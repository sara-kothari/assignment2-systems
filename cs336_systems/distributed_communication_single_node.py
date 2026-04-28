import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
import timeit
import numpy as np
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] ="localhost"
    os.environ["MASTER_PORT"] ="29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def distributed_demo(rank, world_size, data_size):
    
    
    setup(rank, world_size)
    data = torch.randint(0, 10, (data_size,), device="cuda",dtype=torch.float32)
    
    for warmup in range(5):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        
    start = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    timings = [None for i in range(world_size)]
    dist.all_gather_object(timings, end-start)
    if rank == 0:
        print(data_size, world_size, np.mean(timings))

def main():
    results = []
    data_sizes = [10**6//4,10**7//4, 10**8//4,10**9//4 ]
    world_sizes = [2,4,6]
    for data_size in data_sizes:
        for world_size in world_sizes:
            
            start = timeit.default_timer()
            mp.spawn(fn=distributed_demo, args=(world_size, data_size,), nprocs=world_size, join=True)
            
            # cur_result = {"data_size": data_size, "world_size": world_size,"runtime": end - start }
            # results.append(cur_result)
    return results
            
            
if __name__ =="__main__":
    main()