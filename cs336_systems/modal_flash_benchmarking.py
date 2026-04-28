import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
import pandas as pd


wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=7200
)
def benchmark():
    from cs336_systems.flash_benchmarking import main
    import torch
    results = main()
    df = pd.DataFrame(results)
    df.to_csv(f"data/results_fa_v_torch.csv", mode="a", index=False)
    

@app.local_entrypoint()
def modal_main():
    # results = benchmark.remote()
    # df = pd.DataFrame(results)
    # output_dir = "data/flash_attention"
    # os.makedirs(output_dir, exist_ok=True)
    # df.to_csv(f"{output_dir}/results_fa_v_torch.csv", mode="a", index=False)
    benchmark.remote()