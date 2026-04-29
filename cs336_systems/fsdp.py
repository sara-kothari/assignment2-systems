import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.model import *
import time
def shard_param(param, rank, world_size):
        dim0 = param.shape[0]
        assert dim0 % world_size == 0
        shard_size = dim0 // world_size
        start = rank * shard_size
        end = (rank + 1) * shard_size

        return param.data[start:end].clone()


class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.handles = []
        self.module = module
        self.compute_dype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sharded_layers = []
        
        #broadcast the data to ensure same initialization
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            
        
        def all_gather_forward_pre_hook(layer, inputs):
            if (layer.all_gathered):
                    return
            layer.all_gathered = True
            full= torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=torch.float32)
            dist.all_gather_into_tensor(full, layer.weight.data)
            layer.original_shard = layer.weight.data
            if compute_dtype is not None:
                full = full.to(compute_dtype)
            layer.weight.data = full
        #     print("okay gather")
            if compute_dtype is not None and not isinstance(layer, (Embedding, nn.Embedding)):
                return (inputs[0].to(compute_dtype),)
        
        def all_gather_post_hook(idx):
                def hook(layer, inputs, output):
                        if idx + 2 < len(self.sharded_layers):
                                target = self.sharded_layers[idx + 2]
                                all_gather_forward_pre_hook(target, inputs)
                return hook
                
        
        cur_index = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                shard = shard_param(layer.weight,self.rank, self.world_size)
                layer.weight = nn.Parameter(shard)
                layer.all_gathered = False
                self.sharded_layers.append(layer)
                layer.register_forward_pre_hook(all_gather_forward_pre_hook)
                layer.register_forward_hook(all_gather_post_hook(cur_index))
                cur_index +=1
        
    def forward(self,*inputs,**kwargs):
        for layer in self.sharded_layers:
                layer.all_gathered = False
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        
        def reduce_scatter_backward_post_hook(layer, inputs, outputs):
            if layer.weight.grad is None:
                layer.weight.data = layer.original_shard
                del layer.original_shard
                # print("no grad")
                return 
            full_grad = layer.weight.grad
            shard_grad = torch.empty(full_grad.shape[0]//self.world_size, *full_grad.shape[1:], device=layer.weight.device, dtype=layer.weight.dtype)
            dist.reduce_scatter_tensor(shard_grad, full_grad)
            layer.weight.data = layer.original_shard
            del layer.original_shard
        #     print("layer weight", layer.weight.data.shape)
        #     print("layer", layer.weight.grad.shape)
        #     print(shard_grad.shape)
            shard_grad = shard_grad / self.world_size
            layer.weight.grad = shard_grad.to(torch.float32)
        #     print("okay reduce")
            
        for layer in self.sharded_layers:
            
            if layer.weight.grad is None:
                # print("no grad in sync")
                continue
            
            reduce_scatter_backward_post_hook(layer, None, None)
        #     print("layer weight data in sync", layer.weight.data.shape, "layer grad shape", layer.weight.grad.shape)
        
        
        sharded_params = set(id(l.weight) for l in self.sharded_layers)

        for param in self.module.parameters():
                if id(param) not in sharded_params and param.grad is not None:
                        dist.all_reduce(param.grad.data)
                        param.grad.data /= self.world_size
    
    def fsdp_gather_full_params(self):
        state_dict = {}
        for name, layer in self.module.named_modules():
            if name == "":
                continue
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                full_weight=torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=layer.weight.dtype)
                dist.all_gather_into_tensor(full_weight,layer.weight.data, async_op=False)
                state_dict[name + ".weight"] = full_weight
            else:
                if hasattr(layer, 'weight'):
                    state_dict[name + ".weight"] = layer.weight
        
        return state_dict
                