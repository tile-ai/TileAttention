# Copyright (c) Tile-AI.
# Licensed under the MIT License.

import argparse
from top import LinearAttentionFusedRecurrentKernel


def test_fused_recurrent(B, S, H, D):
    kernel = LinearAttentionFusedRecurrentKernel(B, S, H, D)
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=2048, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=256, help='head dim')
    args = parser.parse_args()
    B, S, H, D = args.batch, args.seq_len, args.heads, args.dim

    test_fused_recurrent(B, S, H, D)