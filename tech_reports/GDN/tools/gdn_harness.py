# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""GDN kernel harness: run ttnn.transformer.chunk_gated_delta_rule on the open device (silicon or ttsim),
dump every output so two kernel builds can be compared bit for bit, and report PCC against a torch golden.

Usage: gdn_harness.py --tag NAME [--mode flat|split] [--path fused|phased] [--bh N] [--hk N] [--t N]
                      [--nv N] [--np N] [--wy auto|horner|fs] [--chunk 32|64] [--reps N] [--hard] [--seed N]
                      [--compare OTHER_TAG]
Outputs land in $GDN_OUT/<tag>.pt (default: the current directory). See ../chunk_gdn_kernel_optimization.md.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

import ttnn

CHUNK, KDIM, VDIM = 32, 128, 128


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


def golden(q, k, v, g, beta, scale, s0, C, qk_norm):
    """Whole-computation torch golden (the test suite's _golden_chunk_gdn, plus the in-kernel qk-norm form)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if G > 1:
        q = q.repeat_interleave(G, dim=2)
        k = k.repeat_interleave(G, dim=2)
    if qk_norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    else:
        q = (q * scale).to(torch.bfloat16).to(torch.float32)
    BH, NC = B * HV, T // C
    q_c, k_c, v_c = (x.permute(0, 2, 1, 3).reshape(BH, NC, C, x.shape[-1]) for x in (q, k, v))
    g_c, beta_c = (x.permute(0, 2, 1).reshape(BH, NC, C) for x in (g, beta))
    decay = g_c.cumsum(-1)
    decay_exp = decay.exp().unsqueeze(-1)
    v_beta = v_c * beta_c.unsqueeze(-1)
    k_beta = k_c * beta_c.unsqueeze(-1)
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    mask_upper = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k_c.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(C, dtype=torch.float32)
    kd = k_beta * decay_exp
    q_decay = q_c * decay_exp
    mask_causal = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    intra = (q_c @ k_c.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)
    k_dec_t = (k_c * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)
    dl = decay[..., -1].exp()
    S = s0.reshape(BH, K, V).clone() if s0 is not None else torch.zeros(BH, K, V, dtype=torch.float32)
    o = torch.zeros(BH, NC, C, V, dtype=torch.float32)
    for c in range(NC):
        v_new = t_inv[:, c] @ (v_beta[:, c] - kd[:, c] @ S)
        o[:, c] = q_decay[:, c] @ S + intra[:, c] @ v_new
        S = S * dl[:, c, None, None] + k_dec_t[:, c] @ v_new
    o = o.reshape(B, HV, T, V).permute(0, 2, 1, 3).contiguous()
    return o, S.reshape(B, HV, K, V)


def const_tiles(device):
    c = CHUNK
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
    ap.add_argument("--mode", default="flat", choices=["flat", "split"])
    ap.add_argument("--path", default="fused", choices=["fused", "phased"])
    ap.add_argument("--bh", type=int, default=4, help="value heads (B=1)")
    ap.add_argument("--hk", type=int, default=2, help="key heads")
    ap.add_argument("--t", type=int, default=256)
    ap.add_argument("--nv", type=int, default=0)
    ap.add_argument("--np", type=int, default=0)
    ap.add_argument("--wy", default="auto", choices=["auto", "horner", "fs"])
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--hard", action="store_true", help="harder numerics: larger |g|, beta near 1")
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()
    global CHUNK
    CHUNK = args.chunk

    out_dir = os.environ.get("GDN_OUT", ".")
    torch.manual_seed(args.seed)
    B, T, H, HV = 1, args.t, args.hk, args.bh
    flat = args.mode == "flat"
    if flat:
        # production form: raw (un-normalized) flat q/k, normalized in-kernel
        q = (torch.randn(B, T, H, KDIM) * 0.7).to(torch.bfloat16)
        k = (torch.randn(B, T, H, KDIM) * 1.3).to(torch.bfloat16)
    else:
        q = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
        k = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, HV, VDIM)).to(torch.bfloat16)
    if args.hard:
        beta = torch.sigmoid(torch.randn(B, T, HV) * 2.0 + 1.0)
        g = -F.softplus(torch.randn(B, T, HV) * 1.5)
    else:
        beta = torch.sigmoid(torch.randn(B, T, HV))
        g = -F.softplus(torch.randn(B, T, HV)) * 0.5
    s0 = 0.05 * torch.randn(B, HV, KDIM, VDIM)

    device = ttnn.open_device(device_id=0)
    try:

        def dev(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        if flat:
            tq = dev(q.reshape(B, T, H * KDIM), ttnn.bfloat16)
            tk = dev(k.reshape(B, T, H * KDIM), ttnn.bfloat16)
            tv = dev(v.reshape(B, T, HV * VDIM), ttnn.bfloat16)
        else:
            tq, tk, tv = dev(q, ttnn.bfloat16), dev(k, ttnn.bfloat16), dev(v, ttnn.bfloat16)
        tg, tb = dev(g, ttnn.float32), dev(beta, ttnn.float32)
        ts0 = dev(s0, ttnn.float32)
        eye, tril, ones, masks = const_tiles(device)
        if args.path == "fused":
            cfg = ttnn.ChunkGdnFusedProgramConfig(num_receivers=args.nv or None, num_producers=args.np or None)
        else:
            cfg = ttnn.ChunkGdnPhasedProgramConfig()
        wy = {
            "auto": ttnn.ChunkGdnWyInverse.AUTO,
            "horner": ttnn.ChunkGdnWyInverse.HORNER,
            "fs": ttnn.ChunkGdnWyInverse.FORWARD_SUBSTITUTION,
        }[args.wy]
        scale = KDIM**-0.5
        res = None
        for r in range(args.reps):
            t0 = time.time()
            o, fs = ttnn.transformer.chunk_gated_delta_rule(
                tq,
                tk,
                tv,
                tg,
                tb,
                initial_state=ts0,
                output_final_state=True,
                chunk_size=CHUNK,
                program_config=cfg,
                wy_inverse=wy,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
                output_head_major=flat,
            )
            o_t, fs_t = ttnn.to_torch(o).float(), ttnn.to_torch(fs).float()
            ttnn.deallocate(o)
            ttnn.deallocate(fs)
            dt = time.time() - t0
            if res is not None:
                same = torch.equal(res[0], o_t) and torch.equal(res[1], fs_t)
                print(f"rep {r}: {dt*1e3:.1f} ms host, identical to rep 0: {same}")
            else:
                res = (o_t, fs_t)
                print(f"rep {r}: {dt*1e3:.1f} ms host (includes compile)")
        o_t, fs_t = res
        if flat:  # head-major [BH,T,V] -> [B,T,HV,V]
            o_t = o_t.reshape(B, HV, T, VDIM).permute(0, 2, 1, 3).contiguous()
        o_ref, fs_ref = golden(q.float(), k.float(), v.float(), g, beta, scale, s0, CHUNK, flat)
        print(
            f"RESULT tag={args.tag} arch={device.arch()} path={args.path} mode={args.mode} wy={args.wy} "
            f"BH={HV} Hk={H} T={T} pcc_o={pcc(o_ref, o_t):.8f} pcc_fs={pcc(fs_ref, fs_t):.8f} "
            f"maxabs_o={(o_ref - o_t).abs().max().item():.3e} maxabs_fs={(fs_ref - fs_t).abs().max().item():.3e}"
        )
        torch.save({"o": o_t, "fs": fs_t}, os.path.join(out_dir, args.tag + ".pt"))
        if args.compare:
            other = torch.load(os.path.join(out_dir, args.compare + ".pt"))
            eq_o, eq_fs = torch.equal(other["o"], o_t), torch.equal(other["fs"], fs_t)
            # bitwise comparison also distinguishes +0/-0 and NaN payloads
            bit_o = torch.equal(other["o"].view(torch.int32), o_t.view(torch.int32))
            bit_fs = torch.equal(other["fs"].view(torch.int32), fs_t.view(torch.int32))
            print(
                f"COMPARE {args.tag} vs {args.compare}: o equal={eq_o} bitwise={bit_o}; fs equal={eq_fs} bitwise={bit_fs}"
            )
            if not (bit_o and bit_fs):
                d = (other["o"] - o_t).abs()
                print(f"  o: {int((d > 0).sum())} of {d.numel()} differ, max abs diff {d.max().item():.3e}")
                d = (other["fs"] - fs_t).abs()
                print(f"  fs: {int((d > 0).sum())} of {d.numel()} differ, max abs diff {d.max().item():.3e}")
                sys.exit(3)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
