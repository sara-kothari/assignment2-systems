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

def tensor_bytes(t):
    return t.numel() * t.element_size()

def param_bytes(model):
    return sum(tensor_bytes(p) for p in model.parameters())

def grad_bytes(model):
    return sum(tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)

def optimizer_state_bytes(opt):
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += tensor_bytes(v)
    return total

def gb(x):
    return x / (1024**3)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    
    
def distributed_training(rank, world_size, config):
    setup(rank, world_size)
    torch.manual_seed(1)
    data = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]), device=config["device"])
    targets = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]),device=config["device"])
    device_batch_size = config["batch_size"]// world_size
    my_data = data[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    my_targets = targets[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    
    
    #unsharded
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    model = TransformerLM(config["vocab_size"], config["context_length"],
                          config["d_model"], config["num_layers"], 
                          config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to("cuda")
    un_sharded_optimizer = AdamW(model.parameters(), lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,)
    print("INIT")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(un_sharded_optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/unsharded_model_init{rank}.pickle")
    un_sharded_optimizer.zero_grad(set_to_none=True)
    res = model(my_data)
    loss = cross_entropy_loss(res, my_targets)
    loss.backward()
    
    print("PRESTEP")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(un_sharded_optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/unsharded_prestep{rank}.pickle")
    un_sharded_optimizer.step()
    
    print("POSTSTEP")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(un_sharded_optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/unsharded_poststep{rank}.pickle")

    torch.cuda.memory._record_memory_history(enabled=None)
    
    
    # #sharded
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    model = TransformerLM(config["vocab_size"], config["context_length"],
                          config["d_model"], config["num_layers"], 
                          config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to("cuda")
    sharded_optimizer = OSS(
        model.parameters(),
        AdamW,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    print("INIT")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(sharded_optimizer.optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/sharded_model_init{rank}.pickle")
    sharded_optimizer.zero_grad(set_to_none=True)
    res = model(my_data)
    loss = cross_entropy_loss(res, my_targets)
    loss.backward()
    print("PRESTEP")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(sharded_optimizer.optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/sharded_prestep{rank}.pickle")
    sharded_optimizer.step()
    print("POSTSTEP")
    print("params:", gb(param_bytes(model)))
    print("grads:", gb(grad_bytes(model)))
    print("opt:", gb(optimizer_state_bytes(sharded_optimizer.optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/sharded_poststep{rank}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
  
    
    
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