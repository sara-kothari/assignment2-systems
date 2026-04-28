import cs336_basics.transformer as transformer
from cs336_basics.transformer import *
from cs336_basics.training import *
import argparse
import torch
import json 
import timeit 
import pandas as pd
import numpy as np
import os
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext



def main():
    compiled_attn = torch.compile(scaled_dot_product_attention)
    results = []
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384, 32768 ]
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            Q = torch.randn( (8, seq_len, d_model), device="cuda", requires_grad=True)
            K = torch.randn( (8, seq_len, d_model), device="cuda",requires_grad=True)
            V = torch.randn( (8, seq_len, d_model), device="cuda",requires_grad=True)
            print(d_model, seq_len)
            #warmup
            try:
                for i in range(5):
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                    logits = compiled_attn(Q,K,V,None)
                    total = logits.sum()
                    total.backward()
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "fwd_time": "OOM",
                    "bwd_time": "OOM",
                    "memory": "OOM"
                })
                torch.cuda.empty_cache()
                continue
            
            forward_timings = []
            for i in range(100):
                start = timeit.default_timer()
                logits =compiled_attn(Q,K,V,None)
                torch.cuda.synchronize()
                end = timeit.default_timer()
                forward_timings.append(end - start)
            memory = torch.cuda.memory_allocated()
            
            backward_timings = []
            for i in range(100):
                Q.grad = None
                K.grad = None
                V.grad = None
                logits =compiled_attn(Q,K,V,None)
                torch.cuda.synchronize()
                start = timeit.default_timer()
                total = logits.sum()
                total.backward()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                backward_timings.append(end - start)
            
            results.append ({
            "d_model": d_model, 
            "seq_len": seq_len,
            "fwd_time": np.mean(forward_timings),
            "bwd_time": np.mean(backward_timings),
            "memory":memory})
            
    return results
if __name__ == "__main__":
    main()
    
        