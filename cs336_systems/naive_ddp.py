import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *

class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    
def distributed_training(rank, world_size):
    setup(rank, world_size)
    model = ToyModel()
    # model.to("cuda")
    
    #broadcast model params
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optimizer = AdamW(model.parameters(), (0.9, 0.95), 1e-8, 0.1, 1e-3)
    my_dict = {}
    # save my_dict to initialize the single device training params
    if rank == 0:
        my_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    
    #create data
    torch.manual_seed(1)
    batch_size = 32
    dim = 10
    data = torch.randn((batch_size, dim))
    assert batch_size % world_size == 0
    device_batch_size = batch_size// world_size
    my_data = data[rank*device_batch_size : (rank+1)*device_batch_size]
    # my_data = my_data.to("cuda")
    
    #training
    for step in range(15):
        optimizer.zero_grad()
        out = model(my_data)
        loss = out.sum()
        loss.backward()
        
        #all reduce the params
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data)
                param.grad.data = param.grad.data/world_size
        optimizer.step()
    
    if rank == 0:
        single_device_model = single_device_training(data, 15, my_dict)
        for param1, param2 in zip(model.parameters(), single_device_model.parameters()):
            # assert torch.equal(param1.data, param2.data) == True
            assert torch.allclose(param1, param2, atol=1e-6) == True
        print("all params match")
                

def single_device_training(x, num_steps, my_dict):
    model = ToyModel()
    model.load_state_dict(my_dict["model"])
    # model.to("cuda")
    optimizer = AdamW(model.parameters(), (0.9, 0.95), 1e-8, 0.1, 1e-3)
    optimizer.load_state_dict(my_dict["optimizer"])
    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    return model
    
def main():
    world_size = 4
    mp.spawn(fn=distributed_training, args=(world_size,), nprocs=world_size, join=True)
            
if __name__ =="__main__":
    main()
