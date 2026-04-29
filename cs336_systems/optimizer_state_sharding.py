import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.transformer import *


class OSS(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls,**kwargs):
        self.world_size = dist.get_world_size()
        self.cur_rank = 0
        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self.param_to_rank = {}
        self.all_params = []
        self.kwargs = kwargs
        super().__init__(params, defaults=kwargs)
       
    def step(self, closure=None,**kwargs):
        self.optimizer.step(closure, **kwargs)
        for i in range(len(self.all_params)):
            param = self.all_params[i]
            dist.broadcast(param.data, src=self.param_to_rank[i])
            
    def add_param_group(self, param_group): 
        device_params = []
        for param in param_group["params"]:
            if ((self.cur_rank) % self.world_size) == dist.get_rank():
                device_params.append(param)
            self.all_params.append(param)
            self.param_to_rank[len(self.all_params)-1] = self.cur_rank % self.world_size
            self.cur_rank += 1
        device_group = {**param_group, "params": device_params}
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([device_group], **self.kwargs)
        else:
            self.optimizer.add_param_group(device_group)
        super().add_param_group(param_group)


