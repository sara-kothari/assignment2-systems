import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.model import BasicsTransformerLM, Linear, Embedding
# from cs336_basics.transformer import Linear, Embedding
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        print("in FSDP")
        #broadcast the data to ensure same initialization
        # for param in self.module.parameters():
        #     dist.broadcast(param.data, src=0)
            
        
        def all_gather_forward_pre_hook(layer, inputs):
        #     if (layer.all_gathered):
        #             return
        #     layer.all_gathered = True
               
                if hasattr(layer, 'original_shard'):
                        return
                full= torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=torch.float32)
                dist.all_gather_into_tensor(full, layer.weight.data)
                layer.original_shard = layer.weight.data
                if compute_dtype is not None:
                        full = full.to(compute_dtype)
                layer.weight.data = full
                if compute_dtype is not None and not isinstance(layer, (Embedding, nn.Embedding)):
                        return (inputs[0].to(compute_dtype),)
        
        def restore_shard_forward_post_hook(layer, inputs, output):
                # print(f"restoring, freeing {layer.weight.data.nbytes / 1e9:.2f} GB")
                layer.weight.data = layer.original_shard
                del layer.original_shard
        
        def all_gather_backward_pre_hook(layer, grad_output):
                # dtype = compute_dtype if compute_dtype else layer.weight.dtype
                # full = torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=dtype)
                # dist.all_gather_into_tensor(full, layer.weight.data)
                # layer.original_shard = layer.weight.data
                # # if compute_dtype is not None:
                # #         full = full.to(compute_dtype)
                # layer.weight.data = full
                full = torch.empty(
                        layer.weight.shape[0]*self.world_size, 
                        *layer.weight.shape[1:], 
                        device=layer.weight.device, 
                        dtype=layer.weight.dtype  # match the actual weight dtype
                )
                dist.all_gather_into_tensor(full, layer.weight.data)
                layer.original_shard = layer.weight.data
                if compute_dtype is not None:
                        full = full.to(compute_dtype)
                layer.weight.data = full
        
        def all_gather_post_hook(idx):
                def hook(layer, inputs, output):
                        if idx + 2 < len(self.sharded_layers):
                                target = self.sharded_layers[idx + 2]
                                all_gather_forward_pre_hook(target, inputs)
                return hook
        
       
        def reduce_scatter_hook_async(layer):
                def hook(param):
                        if param.grad is None:
                                print("no grad")
                                return
                        full_grad = param.grad
                        shard_grad = torch.empty(full_grad.shape[0]//self.world_size, *full_grad.shape[1:], device=layer.weight.device, dtype=layer.weight.dtype)
                        handle = dist.reduce_scatter_tensor(shard_grad, full_grad, async_op=True)
                        self.handles.append((handle, param, shard_grad, layer))
                        param.grad = None
                       
                return hook

                        
        print("adding hooks")        
        cur_index = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                # print(name)
                shard = shard_param(layer.weight,self.rank, self.world_size)
                layer.weight = nn.Parameter(shard)
                layer.all_gathered = False
                self.sharded_layers.append(layer)
                layer.register_forward_pre_hook(all_gather_forward_pre_hook)
                layer.register_forward_hook(all_gather_post_hook(cur_index))
                layer.register_forward_hook(restore_shard_forward_post_hook)
                layer.register_full_backward_pre_hook(all_gather_backward_pre_hook)
                layer.weight.register_post_accumulate_grad_hook(reduce_scatter_hook_async(layer))
                cur_index +=1
        print("len sharded", len(self.sharded_layers))
        
    def forward(self,*inputs,**kwargs):
        for layer in self.sharded_layers:
                layer.all_gathered = False
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, param, shard_grad, layer in self.handles:
            handle.wait()
            layer.weight.data = layer.original_shard
            del layer.original_shard
            shard_grad = shard_grad / self.world_size
            param.grad = shard_grad.to(torch.float32)
        self.handles.clear()
        
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
                