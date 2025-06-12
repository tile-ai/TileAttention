# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import tilelang as tl
from tilelang.profiler import do_bench
import tilelang.language as T
import fla.ops.linear_attn # We compare with Triton implementation in FLA


@tl.jit(out_idx=[3])
def _fused_recurrent_fwd(B,
                     S,
                     H,
                     D,
                     scale=None,
                     dtype='float16',
                     BK=32,
                     BV=32):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV

    if scale is None:
        scale = D ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, H, D], dtype), # type: ignore
        K: T.Tensor([B, S, H, D], dtype), # type: ignore
        V: T.Tensor([B, S, H, D], dtype), # type: ignore
        Output: T.Tensor([NK, B, S, H, D], dtype) # type: ignore
    ): 
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([BK], accum_dtype)
            k = T.alloc_shared([BK], accum_dtype)
            v = T.alloc_shared([BV], accum_dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            o = T.alloc_fragment([BV, BK], accum_dtype)
            o_sum = T.alloc_fragment([BV], accum_dtype)
            T.clear(h)

            for i in T.serial(0, S):
                for j in T.Parallel(BK):
                    q[j] = Q[i_b, i, i_h, i_k * BK + j] * scale
                T.copy(K[i_b, i, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i, i_h, i_v * BV:(i_v + 1) * BV], v)

                for row, col in T.Parallel(BV, BK):
                    h[row, col] += k[col] * v[row]
                    o[row, col] = h[row, col] * q[col]
                T.reduce_sum(o, o_sum, dim=1)

                T.copy(o_sum, Output[i_k, i_b, i, i_h, i_v * BV:(i_v + 1) * BV])

    return main


@tl.jit(out_idx=[4, 5, 6])
def _fused_recurrent_bwd(B,
                     S,
                     H,
                     D,
                     scale=None,
                     dtype='float16',
                     BK=32,
                     BV=32):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV

    if scale is None:
        scale = D ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, H, D], dtype), # type: ignore
        K: T.Tensor([B, S, H, D], dtype), # type: ignore
        V: T.Tensor([B, S, H, D], dtype), # type: ignore
        dO: T.Tensor([B, S, H, D], dtype), # type: ignore
        dQ: T.Tensor([NV, B, S, H, D], dtype), # type: ignore
        dK: T.Tensor([NV, B, S, H, D], dtype), # type: ignore
        dV: T.Tensor([NK, B, S, H, D], dtype), # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            
            q = T.alloc_shared([BK], accum_dtype)
            k = T.alloc_shared([BK], accum_dtype)
            v = T.alloc_shared([BV], accum_dtype)
            do = T.alloc_shared([BV], accum_dtype)
            dq = T.alloc_fragment([BK, BV], accum_dtype)
            dq_sum = T.alloc_fragment([BK], accum_dtype)
            dk = T.alloc_fragment([BK, BV], accum_dtype)
            dk_sum = T.alloc_fragment([BK], accum_dtype)
            dv = T.alloc_fragment([BK, BV], accum_dtype)
            dv_sum = T.alloc_fragment([BV], accum_dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            T.clear(h)
            T.clear(dh)
            
            for i in T.serial(0, S):
                T.copy(K[i_b, i, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i, i_h, i_v * BV:(i_v + 1) * BV], v)
                T.copy(dO[i_b, i, i_h, i_v * BV:(i_v + 1) * BV], do)
                
                for row, col in T.Parallel(BK, BV):
                    h[row, col] += k[row] * v[col]
                    dq[row, col] = h[row, col] * do[col] * scale
                T.reduce_sum(dq, dq_sum, dim=1, clear=True)    
                
                T.copy(dq_sum, dQ[i_v, i_b, i, i_h, i_k * BK:(i_k + 1) * BK])  
            
            for i in T.serial(0, S):
                start = S - 1 - i
                T.copy(K[i_b, start, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, start, i_h, i_v * BV:(i_v + 1) * BV], v)
                T.copy(dO[i_b, start, i_h, i_v * BV:(i_v + 1) * BV], do)
                for j in T.Parallel(BK):
                    q[j] = Q[i_b, start, i_h, i_k * BK + j] * scale
                
                for row, col in T.Parallel(BK, BV):
                    dh[row, col] += q[row] * do[col]
                    dk[row, col] = dh[row, col] * v[col]
                    dv[row, col] = dh[row, col] * k[row]
                T.reduce_sum(dk, dk_sum, dim=1, clear=True)
                T.reduce_sum(dv, dv_sum, dim=0, clear=True)
                
                T.copy(dk_sum, dK[i_v, i_b, start, i_h, i_k * BK:(i_k + 1) * BK])
                T.copy(dv_sum, dV[i_k, i_b, start, i_h, i_v * BV:(i_v + 1) * BV])
         
    return main


class _fused_recurrent_linear_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale, dtype, BK, BV):
        B, S, H, D = q.shape
        ctx.B, ctx.S, ctx.H, ctx.D, ctx.scale, ctx.dtype, ctx.BK, ctx.BV = (
            B, S, H, D, scale, dtype, BK, BV)
        ctx.save_for_backward(q, k, v)

        mod = _fused_recurrent_fwd(B, S, H, D, scale, dtype, BK, BV)
        o = mod(q, k, v)
        return o[0] if o.size(0) == 1 else o.sum(0)

    @staticmethod
    def backward(ctx, do):
        B, S, H, D, scale, dtype, BK, BV, = (ctx.B, ctx.S, ctx.H,
                                                        ctx.D, ctx.scale,
                                                        ctx.dtype, ctx.BK,
                                                        ctx.BV)
        q, k, v = ctx.saved_tensors

        mod = _fused_recurrent_bwd(B, S, H, D, scale, dtype, BK, BV)
        dq, dk, dv = mod(q, k, v, do)
        dq = dq[0] if dq.size(0) == 1 else dq.sum(0)
        dk = dk[0] if dk.size(0) == 1 else dk.sum(0)
        dv = dv[0] if dv.size(0) == 1 else dv.sum(0)
        return dq, dk, dv, None, None, None, None


fused_recurrent_linear_attention = _fused_recurrent_linear_attention.apply


class LinearAttentionFusedRecurrentKernel(nn.Module):

    def __init__(self,
                 batch_size,
                 seq_len,
                 num_heads,
                 head_dim,
                 dtype='float16',
                 block_K=32,
                 block_V=32):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }[dtype]
        self.block_K = block_K
        self.block_V = block_V

        self.attention = fused_recurrent_linear_attention

    def forward(self, q, k, v, scale=None):  # Layout: [B, S, H, D]
        return self.attention(q, k, v, scale, self.dtype, self.block_K,
                              self.block_V)

    def ref_program(self, q, k, v, scale=None):
        return fla.ops.linear_attn.fused_recurrent_linear_attn(q,
                                                           k,
                                                           v,
                                                           scale,
                                                           normalize=False)

    def gen_inputs(self, n: int):
        return (torch.randn(
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
            device='cuda',
            dtype=self._dtype,
            requires_grad=True) for _ in range(n))

    def profile(self, warmup=100):
        q, k, v, do = self.gen_inputs(4)
        # fwd
        with torch.no_grad():
            fwd_latency = do_bench(lambda: self.forward(q, k, v),
                                   warmup=warmup)
            print(f"Fwd latency: {fwd_latency:.2f} ms")
            fwd_ref_latency = do_bench(lambda: self.ref_program(q, k, v),
                                       warmup=warmup)
            print(f"Fwd ref latency: {fwd_ref_latency:.2f} ms")
        # bwd
        o = self.forward(q, k, v)
        bwd_latency = do_bench(
            lambda: o.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd latency: {bwd_latency:.2f} ms")
        o_ref, _ = self.ref_program(q, k, v)
        bwd_ref_latency = do_bench(
            lambda: o_ref.backward(do, retain_graph=True), warmup=warmup)
        print(f"Bwd ref latency: {bwd_ref_latency:.2f} ms")

    def check(self):
        q, k, v, do = self.gen_inputs(4)
        o = self.forward(q, k, v)
        o.backward(do)
        dq, q.grad = q.grad.clone(), None
        dk, k.grad = k.grad.clone(), None
        dv, v.grad = v.grad.clone(), None
        o_ref, _ = self.ref_program(q, k, v)
        o_ref.backward(do)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
        assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2), f"o does not match reference, {torch.max(torch.abs(o - o_ref))}"
        assert torch.allclose(dq, dq_ref, atol=1e-2, rtol=1e-2), "dq does not match reference"
        assert torch.allclose(dk, dk_ref, atol=1e-2, rtol=1e-2), "dk does not match reference"
        assert torch.allclose(dv, dv_ref, atol=1e-2, rtol=1e-2), "dv does not match reference"
        print("All checks passed! ✅")


if __name__ == '__main__':
    kernel = LinearAttentionFusedRecurrentKernel(8, 1024, 32, 256)
    kernel.check()
    kernel.profile()