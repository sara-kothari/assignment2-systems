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
    import os
    import subprocess
    import json
    import tempfile
    import shutil
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
    f"-- python -m cs336_systems.benchmarking --config {config_path}"
    )
    subprocess.run(cmd, shell=True, check=True)
    shutil.copy("/tmp/profile_result.nsys-rep", f"{DATA_PATH}/profile_result.nsys-rep")
    # return main(config)
    return {}

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict = json.load(f)
    results = benchmark.remote(config_dict)
    if results:
        df = pd.DataFrame([results])
        output_dir = "data/2_1_3"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}/results_2_1_3_1warmup.csv", mode="a", header=not os.path.exists(f"{output_dir}/results_2_1_3.csv"), index=False)