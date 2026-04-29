import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
import torch.nn as nn
from cs336_basics.training import *
from cs336_basics.transformer import *



class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.handles = []
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sharded_layers = []
        self.cur_layer = 0
        count = 0
        self.compute_dtype = compute_dtype
        #broadcast each
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        def all_gather_hook(layer,input ):
            full_weight=torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=layer.weight.dtype)
            handle = dist.all_gather_into_tensor(full_weight,layer.weight, async_op=False)
            if compute_dtype is not None:
                full_weight = full_weight.to(compute_dtype)
            layer.weight = nn.Parameter(full_weight)
            self.handles.append(handle)
        
        
            
        # def shard_and_delete(layer, input, output):
        #     old = layer.weight.data
        #     split = layer.weight.shape[0]//self.world_size
        #     layer.weight = nn.Parameter(layer.weight[split*self.rank: split*(self.rank + 1)].to(torch.float32))
        #     del old
        
        
            
        #idk about the timing for this one. 
        def reduce_scatter_hook(layer,g_in, g_out):
            # handle = dist.all_reduce(param.grad.data, async_op=True)
            output = torch.empty((layer.weight.shape[0]//self.world_size, *layer.weight.shape[1:]),device=layer.weight.device, dtype=layer.weight.dtype )
            handle = dist.reduce_scatter_tensor(output, layer.weight.grad, async_op=False)
            layer.weight.grad = output
            self.handles.append(handle)
        
        
        for name, layer in self.module.named_modules():
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                
                self.sharded_layers.append(layer)
                split = layer.weight.shape[0]//self.world_size
                layer.weight = nn.Parameter(layer.weight[split*self.rank: split*(self.rank + 1)])
                layer.register_full_backward_hook(hook=reduce_scatter_hook)
                if (count == 0 or count == 1):
                    layer.register_forward_pre_hook(all_gather_hook)
                    
                def all_gather_general_trigger(layer, input, output,i=count):
                    if i + 2 < len(self.sharded_layers):
                        target_layer = self.sharded_layers[i + 2]
                        full_weight=torch.empty(target_layer.weight.shape[0]*self.world_size, *target_layer.weight.shape[1:], device=target_layer.weight.device, dtype=target_layer.weight.dtype)
                        handle = dist.all_gather_into_tensor(full_weight,target_layer.weight, async_op=False )
                        target_layer.weight = nn.Parameter(full_weight)
                        self.handles.append(handle)

                layer.register_forward_hook(hook=all_gather_general_trigger)
                
                # layer.register_forward_hook(hook=shard_and_delete)
                count +=1
                    
                
    def forward(self,*inputs,**kwargs):
        self.cur_layer = 0
        return self.module.forward(*inputs, **kwargs)
 
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        
        def shard_and_delete(layer):
            old = layer.weight.data
            split = layer.weight.shape[0]//self.world_size
            layer.weight = nn.Parameter(layer.weight[split*self.rank: split*(self.rank + 1)].to(torch.float32))
            del old
        
        for layer in self.sharded_layers:
            shard_and_delete(layer)
            
        
        # for param in self.module.parameters():
        #     if param.grad is not None:
        #         param.grad.data = param.grad.data/dist.get_world_size()
    
    def fsdp_gather_full_params(self):
        state_dict = {}
        for name, layer in self.module.named_modules():
            if name == "":
                continue
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding)):
                full_weight=torch.empty(layer.weight.shape[0]*self.world_size, *layer.weight.shape[1:], device=layer.weight.device, dtype=layer.weight.dtype)
                dist.all_gather_into_tensor(full_weight,layer.weight, async_op=False)
                state_dict[name + ".weight"] = full_weight
            else:
                if hasattr(layer, 'weight'):
                    state_dict[name + ".weight"] = layer.weight
        
        return state_dict
                
                
        