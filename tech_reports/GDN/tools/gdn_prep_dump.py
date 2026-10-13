# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Dump the seven outputs of the private chunk_gdn_prep prim (the prep compute kernel's hand-off tensors) so two
kernel builds can be compared bit for bit at the stage level, and check each against torch.

gdn_prep_dump.py --tag NAME [--chunk 32|64] [--norm] [--wy auto|horner|fs] [--bh N] [--nc N] [--hard] [--compare TAG]
  --norm : flat raw q/k/v, in-kernel L2 norm (the production form; chunk 32 only)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

import ttnn

KDIM, VDIM = 128, 128
NAMES = ["v_beta", "nkd", "q_decay", "intra", "k_dec_t", "dl", "t_inv"]


def pcc(g, a):
    g = g.to(torch.float64).flatten()
    a = a.to(torch.float64).flatten()
    if not torch.isfinite(a).all():
        return float("nan")
    if torch.equal(g, a):
        return 1.0
    vg, va = g - g.mean(), a - a.mean()
    d = vg.norm() * va.norm()
    return float((vg @ va) / d) if d != 0 else 0.0


def prep_reference(q, k, v, g, beta):
    C = q.shape[-2]
    decay = g.cumsum(-1)
    decay_exp = decay.exp().unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    mask_upper = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(C, dtype=torch.float32)
    nkd = -(k_beta * decay_exp)
    q_decay = q * decay_exp
    mask_causal = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    intra = (q @ k.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)
    k_dec_t = (k * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)
    dl = decay[..., -1].exp()[..., None, None] * torch.eye(32, dtype=torch.float32)
    return v_beta, nkd, q_decay, intra, k_dec_t, dl, t_inv


def const_tiles(device, c):
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return up(eye), up(tril), up(ones), up(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--wy", default="auto", choices=["auto", "horner", "fs"])
    ap.add_argument("--bh", type=int, default=4)
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    out_dir = os.environ.get("GDN_OUT", ".")
    C, bh, nc = args.chunk, args.bh, args.nc
    torch.manual_seed(args.seed)
    scale = KDIM**-0.5
    if args.norm:
        q_raw = (torch.randn(bh, nc, C, KDIM) * 0.7).to(torch.bfloat16)
        k_raw = (torch.randn(bh, nc, C, KDIM) * 1.3).to(torch.bfloat16)
        qf, kf = q_raw.float(), k_raw.float()
        q_ref = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * scale
        k_ref = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        q_in, k_in = q_raw, k_raw
    else:
        q_in = (F.normalize(torch.randn(bh, nc, C, KDIM), dim=-1) * scale).to(torch.bfloat16)
        k_in = F.normalize(torch.randn(bh, nc, C, KDIM), dim=-1).to(torch.bfloat16)
        q_ref, k_ref = q_in.float(), k_in.float()
    v = (0.5 * torch.randn(bh, nc, C, VDIM)).to(torch.bfloat16)
    if args.hard:
        beta = torch.sigmoid(torch.randn(bh, nc, C) * 2.0 + 1.0)
        g = -F.softplus(torch.randn(bh, nc, C) * 1.5)
    else:
        beta = torch.sigmoid(torch.randn(bh, nc, C))
        g = -F.softplus(torch.randn(bh, nc, C)) * 0.5

    device = ttnn.open_device(device_id=0)
    try:

        def dev(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        wy = {
            "auto": ttnn.ChunkGdnWyInverse.AUTO,
            "horner": ttnn.ChunkGdnWyInverse.HORNER,
            "fs": ttnn.ChunkGdnWyInverse.FORWARD_SUBSTITUTION,
        }[args.wy]
        kwargs = dict(chunk_size=C, wy_inverse=wy)
        if args.norm:
            # flat token-major [B=1, T, H*D]: head h's chunk c is rows [c*C, +C), columns [h*D, +D)
            T = nc * C

            def flat(x):  # [bh,nc,C,D] -> [1, T, bh*D]
                return x.permute(1, 2, 0, 3).reshape(1, T, bh * x.shape[-1])

            tq, tk, tv = dev(flat(q_in), ttnn.bfloat16), dev(flat(k_in), ttnn.bfloat16), dev(flat(v), ttnn.bfloat16)
            kwargs.update(v_flat=True, HV=bh, qk_norm=True, scale=scale, qk_flat=True, Hk=bh)
        else:
            tq, tk, tv = dev(q_in, ttnn.bfloat16), dev(k_in, ttnn.bfloat16), dev(v, ttnn.bfloat16)
        tg, tb = dev(g.unsqueeze(-1), ttnn.float32), dev(beta.unsqueeze(-1), ttnn.float32)
        outs = ttnn._ttnn.operations.transformer.chunk_gdn_prep(tq, tk, tv, tg, tb, *const_tiles(device, C), **kwargs)
        got = [ttnn.to_torch(o).float() for o in outs]
        refs = prep_reference(q_ref, k_ref, v.float(), g, beta)
        line = " ".join(
            f"{n}:pcc={pcc(r, o):.7f},max={(o - r).abs().max().item():.2e}" for n, o, r in zip(NAMES, got, refs)
        )
        print(f"RESULT tag={args.tag} arch={device.arch()} C={C} norm={args.norm} wy={args.wy} {line}")
        torch.save({n: o for n, o in zip(NAMES, got)}, os.path.join(out_dir, args.tag + ".pt"))
        if args.compare:
            other = torch.load(os.path.join(out_dir, args.compare + ".pt"))
            bad = []
            for n, o in zip(NAMES, got):
                if not torch.equal(other[n].view(torch.int32), o.view(torch.int32)):
                    d = (other[n] - o).abs()
                    nz = int((other[n].view(torch.int32) != o.view(torch.int32)).sum())
                    bad.append(f"{n}: {nz}/{d.numel()} words differ, max abs {d.max().item():.3e}")
            print(
                f"COMPARE {args.tag} vs {args.compare}: "
                + ("ALL 7 BITWISE IDENTICAL" if not bad else "DIFF " + "; ".join(bad))
            )
            if bad:
                sys.exit(3)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
