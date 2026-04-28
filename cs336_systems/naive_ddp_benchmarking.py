import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.transformer import *



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def distributed_training(rank, world_size, config):
    setup(rank, world_size)
    model = TransformerLM(config["vocab_size"], config["context_length"],
                          config["d_model"], config["num_layers"], 
                          config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to("cuda")
    optimizer = AdamW(model.parameters(), (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"], config["lr"])
    
    #broadcast model params
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
        
    #create data
    torch.manual_seed(1)
    data = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]), device=config["device"])
    targets = torch.randint(low=0, high=config["vocab_size"]-1, size=(config["batch_size"], config["context_length"]),device=config["device"])
    assert config["batch_size"] % world_size == 0
    device_batch_size = config["batch_size"]// world_size
    my_data = data[rank*device_batch_size : (rank+1)*device_batch_size]
    my_data = my_data.to("cuda")
    
    my_targets = targets[rank*device_batch_size : (rank+1)*device_batch_size]
    my_targets = my_targets.to("cuda")
    model.train()
    timings = []
    grad_comm_timings = []
    
    #warmup
    for step in range(1,config["warmup_steps"]+1):
        optimizer.zero_grad()
        logits = model(my_data)
        loss = cross_entropy_loss(logits, my_targets)
        loss.backward()
        #unoptmized:
        # for param in model.parameters():
        #     if param.grad is not None:
        #         dist.all_reduce(param.grad.data)
        #         param.grad.data = param.grad.data/world_size
        
        #flatten optimized
        device_grads = [param.grad.data for param in model.parameters() if param.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(device_grads)
        dist.all_reduce(flat_grads)
        flat_grads = flat_grads/world_size
        unflattened = torch._utils._unflatten_dense_tensors(flat_grads, device_grads)
        params_with_grads = [param for param in model.parameters() if param.grad is not None]
        for param,combined in zip(params_with_grads, unflattened):
            if param.grad is not None:
                param.grad.data = combined
        optimizer.step()
    
    
    #training
    for step in range(1, config["total_steps"] + 1):
        start = timeit.default_timer()
        optimizer.zero_grad()
        logits = model(my_data)
        loss = cross_entropy_loss(logits, my_targets)
        loss.backward()
        torch.cuda.synchronize()
        comm_start = timeit.default_timer()
        #unoptimized
        # for param in model.parameters():
        #     if param.grad is not None:
        #         dist.all_reduce(param.grad.data)
        #         param.grad.data = param.grad.data/world_size
        
        #flatten optimized
        device_grads = [param.grad.data for param in model.parameters() if param.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(device_grads)
        dist.all_reduce(flat_grads)
        flat_grads = flat_grads/world_size
        unflattened = torch._utils._unflatten_dense_tensors(flat_grads, device_grads)
        params_with_grads = [param for param in model.parameters() if param.grad is not None]
        for param,combined in zip(params_with_grads, unflattened):
            if param.grad is not None:
                param.grad.data = combined
        
        torch.cuda.synchronize()
        comm_end = timeit.default_timer()
        grad_comm_timings.append(comm_end - comm_start)
        optimizer.step()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        timings.append(end - start)
        
    all_avg_timings = [None for i in range(world_size)]
    dist.all_gather_object(all_avg_timings, np.mean(timings))
    all_avg_comm = [None for i in range(world_size)]
    dist.all_gather_object(all_avg_comm, np.mean(grad_comm_timings))
    if rank == 0:
        print("full step: ", np.mean(all_avg_timings), " comm time: ", np.mean(all_avg_comm))

    
def main(config):
    world_size = 2
    mp.spawn(fn=distributed_training, args=(world_size,config), nprocs=world_size, join=True)
            
if __name__ =="__main__":
    config = {}
    main(config)
