import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.model import *
import cs336_basics.model_provided
from cs336_basics.model_provided import BasicsTransformerLM
from cs336_systems.ddp_class import DDP
import triton
from cs336_systems.fsdp import *
from cs336_systems.flash_attention import *
# class Config:
#     ctx_len = 32768
#     vocab_size = 151936
#     d_model = 4096
#     d_ff = 11008
#     num_layers = 34
#     num_heads = 32
#     torch_dtype = torch.bfloat16
#     is_causal = True
#     batch_size = 2

class Config:
    ctx_len = 256
    vocab_size = 151936
    d_model = 64
    d_ff = 11008
    num_layers = 1
    num_heads = 2
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2
    
cfg = Config()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def fa_forward(Q,K,V, is_causal=True, mask=None):
    return FA2Triton.apply(Q,K,V, True)
cs336_basics.model_provided.scaled_dot_product_attention = fa_forward 

def distributed_training(rank, world_size,config):
    setup(rank, world_size)
    # full_label, full_target = torch.randint(high=config["vocab_size"], size=(2, config["batch_size"],
    # config["context_length"]))

    # split = full_label.shape[0]//world_size
    # labels = full_label[rank*split : (rank+1)*split]
    # targets = full_target[rank*split : (rank+1)*split]
    # model = TransformerLM(config["vocab_size"], config["context_length"],
    #                       config["d_model"], config["num_layers"], 
    #                       config["num_heads"], config["d_ff"], config["rope_theta"])
    full_label, full_target = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size,
    cfg.ctx_len))

    split = full_label.shape[0]//world_size
    labels = full_label[rank*split : (rank+1)*split]
    targets = full_target[rank*split : (rank+1)*split]
    fsdp_model = BasicsTransformerLM(
    cfg.vocab_size,
    cfg.ctx_len,
    cfg.d_model,
    cfg.num_layers,
    cfg.num_heads,
    cfg.d_ff
)
    fsdp_model.to("cuda") 
    labels = labels.to("cuda")
    targets = targets.to("cuda")
    # fsdp_model = FSDP(model)
    
    
    optimizer = AdamW(fsdp_model.parameters(), (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"], config["lr"])
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = fsdp_model(labels)
        loss = cross_entropy_loss(res, targets)
        loss.backward()
        fsdp_model.finish_gradient_synchronization()
        optimizer.step()
    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=5)
    print(timing_results)
    
    
    
def main(config):
    world_size = 1
   
    mp.spawn(fn=distributed_training, args=(world_size,config,), nprocs=world_size, join=True)
    # Q = torch.randn(2, 32768, 64, device="cuda", dtype=torch.bfloat16)
    # K = Q.clone()
    # V = Q.clone()

    # with torch.no_grad():
    #     out = FA2Triton.apply(Q, K, V, True)
    #     print(torch.isnan(out).any(), torch.isinf(out).any())
    #     print("done")
            
if __name__ =="__main__":
    config = {}
    main(config)
