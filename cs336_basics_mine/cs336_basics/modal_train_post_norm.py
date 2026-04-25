import modal
import os
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
from cs336_basics.main_post_norm import *

wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=7200
)
def train(config):
    (DATA_PATH / f"checkpoints/ {config["experiment_name"]}").mkdir(parents=True, exist_ok=True)
    main(config)

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict=json.load(f)
    config_dict["checkpoint_dir"] = str(DATA_PATH / "checkpoints/")
    train.remote(config_dict)