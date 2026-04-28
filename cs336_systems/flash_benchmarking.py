

from cs336_basics.training import *
from cs336_systems.flash_attention import *
import argparse
import torch
import json 
import timeit 
import pandas as pd
import numpy as np
import os
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import triton


def pytorch_forward(Q,K,V, is_causal=True, mask=None):
    logits = einsum(Q, K, " ... q d, ... k d -> ... q k ") /(Q.shape[-1]**0.5)
    if is_causal and mask is not None:
        logits = torch.where(mask, -torch.inf, logits)
    scores = softmax(logits, -1)
    return scores @ V

def fa_forward(Q,K,V, is_causal, mask=None):
    return FA2Triton.apply(Q,K,V, is_causal)
    

def benchmark_forward(fn, Q,K,V, mask=None):
    def run():
        O = fn(Q,K,V, True, mask)
        torch.cuda.synchronize()
    return triton.testing.do_bench(run)
    

def benchmark_backward(fn, Q, K, V, mask=None):
    O = fn(Q,K,V, True, mask)
    grad = torch.ones_like(O)
    def run():
        Q.grad =None
        K.grad=None
        V.grad=None
        O.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
    return triton.testing.do_bench(run)

def benchmark_full(fn, Q, K,V, mask=None):
    def run():
        Q.grad =None
        K.grad=None
        V.grad=None
        O = fn(Q,K,V, True, mask)
        loss = O.sum()
        loss.backward()
        torch.cuda.synchronize()
    return triton.testing.do_bench(run)
    
    
    
def main():
    results = []
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [128, 256, 512 ,1024, 2048, 4096, 8192, 16384, 32768,65536 ]
    d_types = [torch.bfloat16, torch.float32]
    for d_type in d_types:
        for d_model in d_model_list:
            for seq_len in seq_len_list:
               
                
                mask = None
                Q = torch.randn( (1, seq_len, d_model), device="cuda", dtype=d_type, requires_grad=True)
                K = torch.randn( (1, seq_len, d_model), device="cuda",dtype=d_type, requires_grad=True)
                V = torch.randn( (1, seq_len, d_model), device="cuda",dtype=d_type, requires_grad=True)
                print(d_model, seq_len)
                #warmup
                try:
                    q_idx = torch.arange(start=0,end=Q.shape[1] , step=1, out=None, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(start=0,end=K.shape[1] , step=1, out=None, device=Q.device).unsqueeze(0)
                    mask = k_idx > q_idx
                    for i in range(5):
                        logits = pytorch_forward(Q,K,V,True,mask)
                        total = logits.sum()
                        total.backward()
                        Q.grad = None
                        K.grad = None
                        V.grad = None
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                
                try:
                    for i in range(5):
                        logits = fa_forward(Q,K,V,True)
                        total = logits.sum()
                        total.backward()
                        Q.grad = None
                        K.grad = None
                        V.grad = None
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                
                cur_result = {"d_model": d_model, 
                "seq_len": seq_len,
                "dtype": d_type} 
                
                if mask is None:
                    cur_result["torch_fwd"] = "OOM"
                    cur_result["torch_bwd"] = "OOM"
                    cur_result["torch_full"] = "OOM"
                else:
                    try:
                        torch_fwd = benchmark_forward(pytorch_forward, Q,K,V, mask)
                        cur_result["torch_fwd"] = torch_fwd
                    except torch.cuda.OutOfMemoryError:
                        print("OOM")
                        cur_result["torch_fwd"] = "OOM"
                        torch.cuda.empty_cache()
                    
                    try:
                        torch_bwd = benchmark_backward(pytorch_forward, Q,K,V, mask)
                        cur_result["torch_bwd"] = torch_bwd
                    except torch.cuda.OutOfMemoryError:
                        print("OOM")
                        cur_result["torch_bwd"] = "OOM"
                        torch.cuda.empty_cache()
                    
                    try:
                        torch_full = benchmark_full(pytorch_forward, Q,K,V, mask)
                        cur_result["torch_full"] = torch_full
                    except torch.cuda.OutOfMemoryError:
                        print("OOM")
                        cur_result["torch_full"] = "OOM"
                        torch.cuda.empty_cache()
                try:
                    fa2_fwd = benchmark_forward(fa_forward, Q,K,V)
                    cur_result["fa2_fwd"] = fa2_fwd
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    cur_result["fa2_fwd"] = "OOM"
                    torch.cuda.empty_cache()
                try:
                    fa2_bwd = benchmark_backward(fa_forward, Q,K,V)
                    cur_result["fa2_bwd"] = fa2_bwd
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    cur_result["fa2_bwd"] = "OOM"
                    torch.cuda.empty_cache()
                try:
                    fa2_full = benchmark_full(fa_forward, Q,K,V)
                    cur_result["fa2_full"] = fa2_full
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    cur_result["fa2_full"] = "OOM"
                    torch.cuda.empty_cache()
                print(cur_result)
                results.append(cur_result)
                
    return results
if __name__ == "__main__":
    main()
    
        