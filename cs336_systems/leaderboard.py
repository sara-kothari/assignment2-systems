import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import time
import torch.nn as nn
from cs336_basics.training import *
# from cs336_basics.model import *
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, Linear, Embedding
from cs336_systems.ddp_class import DDP
import triton
from cs336_systems.fsdp import *
from cs336_systems.flash_attention import *
from cs336_systems.optimizer_state_sharding import OSS
import argparse
import json
import torch.cuda.nvtx as nvtx
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2

# class Config:
#     ctx_len = 1024        
#     vocab_size = 50000   
#     d_model = 512
#     d_ff = 2048
#     num_layers = 4
#     num_heads = 8
#     torch_dtype = torch.bfloat16
#     is_causal = True
#     batch_size = 2


    
cfg = Config()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# def fa_forward(Q,K,V, is_causal=True, mask=None):
#     return FA2Triton.apply(Q,K,V, True)
# cs336_basics.model.scaled_dot_product_attention = fa_forward 

def distributed_training(rank, world_size,config):
    torch.set_float32_matmul_precision('high')
    setup(rank, world_size)
    full_label, full_target = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size,
    cfg.ctx_len))

    split = full_label.shape[0]//world_size
    labels = full_label[rank*split : (rank+1)*split]
    targets = full_target[rank*split : (rank+1)*split]
    torch.manual_seed(42)
    model = BasicsTransformerLM(
    cfg.vocab_size,
    cfg.ctx_len,
    cfg.d_model,
    cfg.num_layers,
    cfg.num_heads,
    cfg.d_ff
).cuda()
   
    # model.to("cuda", dtype=torch.bfloat16) 
    print("start compiling")
    # model = torch.compile(model)
    print("done complling")
    labels = labels.to("cuda")
    targets = targets.to("cuda")
    fsdp_model = FSDP(model,compute_dtype=torch.bfloat16)

   
    sharded_optimizer = OSS(
        fsdp_model.parameters(),
        AdamW,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # optimizer = AdamW(, )
    def train_step():
        sharded_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = fsdp_model(labels)
            print("done with forward")
            loss = cross_entropy_loss(res, targets)
            print("done with loss")

        loss.backward()
        
        print("done with backward")
        fsdp_model.finish_gradient_synchronization()
        sharded_optimizer.step()
       
    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    print(timing_results)
    
    
    
def main(config):
    world_size = 2
   
    mp.spawn(fn=distributed_training, args=(world_size,config,), nprocs=world_size, join=True)
    # Q = torch.randn(2, 32768, 64, device="cuda", dtype=torch.bfloat16)
    # K = Q.clone()
    # V = Q.clone()

    # with torch.no_grad():
    #     out = FA2Triton.apply(Q, K, V, True)
    #     print(torch.isnan(out).any(), torch.isinf(out).any())
    #     print("done")
            
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    main(config)

