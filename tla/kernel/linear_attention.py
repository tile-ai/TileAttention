# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import Tuple, List

import torch
import tilelang as tl
import tilelang.language as T
import torch.nn as nn


def _fused_chunk_fwd(B, H, S, D, scale, dtype,
                     BK, BV, chunk_size):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
    NT = S // chunk_size

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, H, D], dtype), # type: ignore
        K: T.Tensor([B, S, H, D], dtype), # type: ignore
        V: T.Tensor([B, S, H, D], dtype), # type: ignore
        O: T.Tensor([NK, B, S, H, D], dtype), # type: ignore
        final_state: T.Tensor([B, H, D, D], accum_dtype) # type: ignore
    ):
        with T.Kernel(NV, NK, B*H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            
            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype) 
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)
            
            T.annotate_layout({
                q: tl.layout.make_swizzled_layout(q),
                k: tl.layout.make_swizzled_layout(k),
                v: tl.layout.make_swizzled_layout(v),
                h_shared: tl.layout.make_swizzled_layout(h_shared),
                s_shared: tl.layout.make_swizzled_layout(s_shared),
            })
            T.use_swizzle(8)
            
            for i in T.Pipelined(0, NT, num_stages=1):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i*chunk_size+row, i_h, i_k*BK+col] * scale
                T.copy(K[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)
                    
                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(
                        row >= col, 
                        s[row, col],
                        0
                    )
                    
                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(q, h_shared, o)
                T.gemm(k, v, h, transpose_A=True)
                T.copy(o, O[i_k, i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV])
                
            # Output final state
            T.copy(h, final_state[i_b, i_h, i_k * BK:(i_k+1) * BK, i_v * BV:(i_v+1) * BV])
            
    return main


def _fused_chunk_bwd(B, H, S, D, scale, dtype,
                     BK, BV, chunk_size):
    accum_dtype = "float"
    NK = D // BK
    NV = D // BV
    NT = S // chunk_size
    
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
        with T.Kernel(NV, NK, B*H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            
            ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype) 
            ds_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            do = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            h_shared = T.alloc_shared([BV, BK], dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            dh_shared = T.alloc_shared([BK, BV], dtype)
            T.clear(h)
            T.clear(dh)
            
            T.annotate_layout({
                ds_shared: tl.layout.make_swizzled_layout(ds_shared),
                q: tl.layout.make_swizzled_layout(q),
                k: tl.layout.make_swizzled_layout(k),
                v: tl.layout.make_swizzled_layout(v),
                do: tl.layout.make_swizzled_layout(do),
                h_shared: tl.layout.make_swizzled_layout(h_shared),
                dh_shared: tl.layout.make_swizzled_layout(dh_shared)
            })

            
            # Calculate dQ
            for i in T.Pipelined(0, NT, num_stages=1):
                T.copy(K[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_k * BK:(i_k+1) * BK], k)
                T.copy(V[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV], v)
                T.copy(dO[i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV], do)
                
                T.gemm(do, v, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(
                        row >= col, 
                        ds[row, col],
                        0
                    )
                    
                T.gemm(ds_shared, k, dq, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(do, h_shared, dq)
                T.gemm(v, k, h, transpose_A=True)
                for row, col in T.Parallel(chunk_size, BK):
                    dq[row, col] *= scale
                T.copy(dq, dQ[i_v, i_b, i*chunk_size:(i+1)*chunk_size, i_h, i_k * BK:(i_k+1) * BK])
                
            # Calculate dK, dV (reversely)
            for i in T.Pipelined(1, NT+1, num_stages=1):
                start = NT-i
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, start*chunk_size+row, i_h, i_k*BK+col] * scale
                T.copy(K[i_b, start*chunk_size:(start+1)*chunk_size, i_h, i_k * BK:(i_k+1) * BK], k)
                T.copy(V[i_b, start*chunk_size:(start+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV], v)
                T.copy(dO[i_b, start*chunk_size:(start+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV], do)
                T.copy(dh, dh_shared)
                
                # Calculate dk 
                T.gemm(v, do, ds, transpose_B=True, clear_accum=True) # ds here actually means `s`, but we simply reuse the buffer `ds`
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else( 
                        row <= col, 
                        ds[row, col],
                        0
                    )
                T.gemm(ds_shared, q, dk, clear_accum=True)
                T.gemm(v, dh_shared, dk, transpose_B=True)
                
                # Calculate dv
                T.gemm(k, q, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(
                        row <= col, 
                        ds[row, col],
                        0
                    )
                T.gemm(ds_shared, do, dv, clear_accum=True)
                T.gemm(k, dh_shared, dv)
                
                # Update dh
                T.gemm(q, do, dh, transpose_A=True)
                
                T.copy(dk, dK[i_v, i_b, start*chunk_size:(start+1)*chunk_size, i_h, i_k * BK:(i_k+1) * BK])
                T.copy(dv, dV[i_k, i_b, start*chunk_size:(start+1)*chunk_size, i_h, i_v * BV:(i_v+1) * BV])
                
    return main


class FusedChunk_Kernel(nn.Module):
    '''In the FusedChunk attention kernel, we calculate the results in one pass without materializing intermediate hidden states.'''
    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen: int,
                 dim: int,
                 scale: float | None = None,
                 block_K: int = 64,
                 block_V: int = 64,
                 chunk_size: int = 64,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        
        assert dim % block_K == 0 and dim % block_V == 0, 'dim must be divisible by block_K and block_V'
        assert seqlen % chunk_size == 0, 'seqlen must be divisible by chunk_size'
        if scale is None:
            scale = dim ** -0.5
        assert dtype in [torch.float16, torch.bfloat16, torch.float], 'dtype must be float16 or bfloat16'
        dtype = dtype.__str__().split('.')[-1]  # Convert torch dtype to string
        
        # Prepare fwd kernel
        self.fwd_program = _fused_chunk_fwd(batch, heads, seqlen, dim, scale, dtype, 
                                            block_K, block_V, chunk_size)
        self.fwd_kernel = tl.compile(self.fwd_program, out_idx=[3, 4])
        self.fwd_profiler = self.fwd_kernel.get_profiler(
            tensor_supply_type=tl.TensorSupplyType.Randn)

        # Prepare bwd kernel
        self.bwd_program = _fused_chunk_bwd(batch, heads, seqlen, dim, scale, dtype,
                                            block_K, block_V, chunk_size)
        self.bwd_kernel = tl.compile(self.bwd_program, out_idx=[4, 5, 6])
        self.bwd_profiler = self.bwd_kernel.get_profiler(
            tensor_supply_type=tl.TensorSupplyType.Randn)
        
        
    def forward(self, q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
        o, final_state = self.fwd_kernel(q, k, v)
        o = o[0] if o.size(0) == 1 else o.sum(0)
        return o, final_state
    
    def backward(self, q, k, v, do) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = self.bwd_kernel(q, k, v, do)
        dq = dq[0] if dq.size(0) == 1 else dq.sum(0)
        dk = dk[0] if dk.size(0) == 1 else dk.sum(0)
        dv = dv[0] if dv.size(0) == 1 else dv.sum(0)
        return dq, dk, dv

    def profile(self, warmup=500):
        fwd_latency = self.fwd_profiler.do_bench(warmup=warmup)
        bwd_latency = self.bwd_profiler.do_bench(warmup=warmup)
        return fwd_latency, bwd_latency

    def check(self):
        pass


if __name__ == '__main__':
    B, S, H, D = 8, 2048, 64, 256
    q = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
    k = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
    v = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)

    kernel = FusedChunk_Kernel(B, H, S, D)

    fwd_latency, bwd_latency = kernel.profile()
    print(f'Forward latency: {fwd_latency:.3f} ms')
    print(f'Backward latency: {bwd_latency:.3f} ms')