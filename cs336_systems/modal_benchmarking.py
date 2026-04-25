import modal
import os
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
from cs336_systems.benchmarking import *

wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=7200
)
def benchmark(config):
    (DATA_PATH / f"2_1_3").mkdir(parents=True, exist_ok=True)
    config["dir"] = str(DATA_PATH / f"2_1_3")
    return main(config)

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict = json.load(f)
    results = benchmark.remote(config_dict)
    df = pd.DataFrame([results])
    output_dir = "data/2_1_3"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/results_2_1_3_1warmup.csv", mode="a", header=not os.path.exists(f"{output_dir}/results_2_1_3.csv"), index=False)