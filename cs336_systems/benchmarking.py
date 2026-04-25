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

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q,K,V, mask):
    with nvtx.range("computing attention scores"):
        logits = einsum(Q, K, " ... q d, ... k d -> ... q k ") /(Q.shape[-1]**0.5)
        if mask is not None:
            logits = torch.where(mask, logits, -torch.inf)
    with nvtx.range("computing softmax"):
        scores = softmax(logits, -1)
    with nvtx.range("final matmul"):
        result = scores @ V
    return result

transformer.scaled_dot_product_attention = annotated_scaled_dot_product_attention
def main(config):
    model = TransformerLM(config["vocab_size"], config["context_length"],
                          config["d_model"], config["num_layers"], 
                          config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to(config["device"])
    optimizer = AdamW(model.parameters(), (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"], config["lr"])
    mode = config["mode"]
    model.train()
    data = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]), device=config["device"])
    targets = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]),device=config["device"])
    timings = []
    
    if (mode == "fwd_only"):
        for step in range(1,config["warmup_steps"]+1):
            with torch.inference_mode():
                logits = model(data)
        for step in range(1, config["total_steps"] + 1):
            start = timeit.default_timer()
            with torch.inference_mode():
                with nvtx.range("forward"):
                    logits = model(data)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            timings.append(end - start)
                
    if (mode == "fwd_bwd"):
        model.train()
        for step in range(1,config["warmup_steps"]+1):
            optimizer.zero_grad()
            logits = model(data)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
        for step in range(1, config["total_steps"] + 1):
            optimizer.zero_grad()
            start = timeit.default_timer()
            with nvtx.range("forward"):
                logits = model(data)
                loss = cross_entropy_loss(logits, targets)
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            timings.append(end - start)
    
    if (mode == "fwd_bwd_optim"):
        model.train()
        for step in range(1,config["warmup_steps"]+1):
            optimizer.zero_grad()
            logits = model(data)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
            optimizer.step()
        for step in range(1, config["total_steps"] + 1):
            start = timeit.default_timer()
            optimizer.zero_grad()
            with nvtx.range("forward"):
                logits = model(data)
                loss = cross_entropy_loss(logits, targets)
            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            timings.append(end - start)
    results = {
        "model": config["model"], 
        "mode": mode,
        "mean": np.mean(timings),
        "std": np.std(timings)
        
    }
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)
    
        