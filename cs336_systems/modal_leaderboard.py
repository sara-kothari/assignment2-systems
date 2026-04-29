import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
import pandas as pd



wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200:2",
    secrets=[wandb_secret],
    timeout=7200
)
def benchmark(config):
    from cs336_systems.leaderboard import main
    main(config)

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict = json.load(f)
    benchmark.remote(config_dict)
    