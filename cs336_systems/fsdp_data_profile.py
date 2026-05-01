import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
class Config:
    ctx_len = 512
    vocab_size = 10000
    d_model = 2560
    d_ff = 10240
    num_layers = 32
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 4



    
cfg = Config()

from cs336_systems.fsdp import FSDP
import argparse
import json
import triton
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM, Linear, Embedding

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
    data, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size,
    cfg.ctx_len))
    device_batch_size = config["batch_size"]// world_size
    my_data = data[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    my_targets = targets[rank*device_batch_size : (rank+1)*device_batch_size].to("cuda")
    print("my_data", my_data.shape)
    
    #no FSDP
    # torch.cuda.memory._record_memory_history(max_entries=1000000)
    # model = TransformerLM(config["vocab_size"], config["context_length"],
    #                       config["d_model"], config["num_layers"], 
    #                       config["num_heads"], config["d_ff"], config["rope_theta"])
    # model.to("cuda")
    # optimizer = AdamW(model.parameters(), lr=0.1,
    #     weight_decay=0.1,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,)
    # print("INIT")
    # print("params:", gb(param_bytes(model)))
    # print("grads:", gb(grad_bytes(model)))
    # print("opt:", gb(optimizer_state_bytes(optimizer)))
    # torch.cuda.memory._dump_snapshot(f"data/no_fsdpmodel_init{rank}.pickle")
    # optimizer.zero_grad(set_to_none=True)
    # res = model(my_data)
    # loss = cross_entropy_loss(res, my_targets)
    # loss.backward()
    
    # print("PRESTEP")
    # print("params:", gb(param_bytes(model)))
    # print("grads:", gb(grad_bytes(model)))
    # print("opt:", gb(optimizer_state_bytes(optimizer)))
    # torch.cuda.memory._dump_snapshot(f"data/no_fsdpprestep{rank}.pickle")
    # optimizer.step()
    
    # print("POSTSTEP")
    # print("params:", gb(param_bytes(model)))
    # print("grads:", gb(grad_bytes(model)))
    # print("opt:", gb(optimizer_state_bytes(optimizer)))
    # torch.cuda.memory._dump_snapshot(f"data/no_sdppoststep{rank}.pickle")

    # torch.cuda.memory._record_memory_history(enabled=None)
    
    # torch.cuda.empty_cache()
    # FSDP
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    model = BasicsTransformerLM(
    cfg.vocab_size,
    cfg.ctx_len,
    cfg.d_model,
    cfg.num_layers,
    cfg.num_heads,
    cfg.d_ff
).cuda()

    fsdp_model = FSDP(model, compute_dtype=torch.float32)
    optimizer = AdamW(fsdp_model.parameters(), lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,)
    
    print("INIT")
    print("params:", gb(param_bytes(fsdp_model)))
    print("grads:", gb(grad_bytes(fsdp_model)))
    print("opt:", gb(optimizer_state_bytes(optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/fsdp_model_init{rank}.pickle")
    optimizer.zero_grad(set_to_none=True)
    res = fsdp_model(my_data)
    loss = cross_entropy_loss(res, my_targets)
    loss.backward()
    print("PRESTEP")
    print("params:", gb(param_bytes(fsdp_model)))
    print("grads:", gb(grad_bytes(fsdp_model)))
    print("opt:", gb(optimizer_state_bytes(optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/fsdp_prestep{rank}.pickle")
    optimizer.step()
    print("POSTSTEP")
    print("params:", gb(param_bytes(fsdp_model)))
    print("grads:", gb(grad_bytes(fsdp_model)))
    print("opt:", gb(optimizer_state_bytes(optimizer)))
    torch.cuda.memory._dump_snapshot(f"data/fsdp_poststep{rank}.pickle")
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