import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
import pandas as pd
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import os
# import torch
# import timeit
# import numpy as np
# import time
# import torch.nn as nn
# from cs336_basics.training import *
# # from cs336_basics.model import *
# import cs336_basics.model
# from cs336_basics.model import BasicsTransformerLM, Linear, Embedding
# from cs336_systems.ddp_class import DDP
# import triton
# from cs336_systems.fsdp import *
# from cs336_systems.flash_attention import *
# from cs336_systems.optimizer_state_sharding import OSS
# import argparse
# import json
# import torch.cuda.nvtx as nvtx




@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200:4",
    secrets=[],
    timeout=7200
)
def benchmark(config):
    import torch
    from cs336_systems.leaderboard import main
    main(config)

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict = json.load(f)
    benchmark.remote(config_dict)
    