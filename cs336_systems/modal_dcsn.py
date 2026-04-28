import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
import pandas as pd


wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200:6",
    secrets=[wandb_secret],
    timeout=7200
)
def benchmark():
    from cs336_systems.distributed_communication_single_node import main
    import torch
    results = main()
    df = pd.DataFrame(results)
    df.to_csv(f"data/results_ddp_first.csv", mode="a", index=False)
    print(df.to_latex(index=False))
    

@app.local_entrypoint()
def modal_main():
    benchmark.remote()