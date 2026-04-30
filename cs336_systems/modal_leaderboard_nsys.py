import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
import pandas as pd



wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200:4",
    secrets=[wandb_secret],
    timeout=7200
)
def benchmark(config):
    import os
    import subprocess
    import json
    import tempfile
    import shutil
    import torch.cuda.nvtx as nvtx
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(config, f)
        config_path = f.name

    cmd = (
    "PYTORCH_ALLOC_CONF=backend:cudaMallocAsync "
    "PYTORCH_NO_CUDA_MEMORY_CACHING=1 "
    "nsys profile -o /tmp/profile_result "
    "--trace=cuda,cudnn,cublas,nvtx "
    "--pytorch=autograd-nvtx "
    "--gpu-metrics-devices=0 "
    f"-- python -m cs336_systems.leaderboard_nsys --config {config_path}"
    )
    subprocess.run(cmd, shell=True, check=True)
    shutil.copy("/tmp/profile_result.nsys-rep", f"{DATA_PATH}/profile_leaderboard.nsys-rep")


@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict = json.load(f)
    benchmark.remote(config_dict)
    