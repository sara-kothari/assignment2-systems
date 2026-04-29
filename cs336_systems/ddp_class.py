import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.transformer import *



class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.handles = []
        self.module = module
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        def hook(param):
            handle = dist.all_reduce(param.grad.data, async_op=True)
            self.handles.append(handle)
            
        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook=hook)
            
            
    def forward(self,*inputs,**kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    # def named_parameters(self, *args, **kwargs):
    #     return self.module.named_parameters(*args, **kwargs)
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data/dist.get_world_size()
    
    