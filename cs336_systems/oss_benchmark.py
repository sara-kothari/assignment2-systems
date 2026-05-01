import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.transformer import *
from cs336_systems.optimizer_state_sharding import OSS
import argparse
import json
import triton
import torch.cuda.nvtx as nvtx
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    
def timing(optimizer, model, my_data, my_targets):
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(my_data)
        loss = cross_entropy_loss(res, my_targets)
        loss.backward()
        optimizer.step()
       
    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    print(timing_results)
    return timing_results
    
    
    
def distributed_training(rank, world_size, config):
    setup(rank, world_size)
    model = TransformerLM(config["vocab_size"], config["context_length"],
                          config["d_model"], config["num_layers"], 
                          config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to("cuda")
    
    un_sharded_optimizer = AdamW(model.parameters(), lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,)
    
    sharded_optimizer = OSS(
        model.parameters(),
        AdamW,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    #create data
    torch.manual_seed(1)
    data = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]), device=config["device"])
    targets = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]),device=config["device"])
    device_batch_size = config["batch_size"]// world_size
    my_data = data[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    my_targets = targets[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    model.train()
    unsharded_timings = timing(un_sharded_optimizer, model, my_data, my_targets)
    sharded_timings = timing(sharded_optimizer, model, my_data, my_targets)
    print("unsharded ", unsharded_timings)
    print("sharded ", sharded_timings)
    
    
    
def main(config):
    world_size = 2
    mp.spawn(fn=distributed_training, args=(world_size,config), nprocs=world_size, join=True)
            
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)