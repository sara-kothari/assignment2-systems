import modal
import os
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import json
from cs336_basics.training import *
from cs336_basics.transformer import *

from cs336_basics.bpe import *

wandb_secret = modal.Secret.from_name("wandb")

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=7200
)
def infer(config):
    print(config["context_length"])
    checkpoint = str(DATA_PATH / f"checkpoints/ {config["experiment_name"]}/checkpoint_step_5000.pt")
    model = TransformerLM(config["vocab_size"], config["context_length"],config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"])
    optimizer = AdamW(model.parameters(), (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"], config["lr"])

    print("loaded model")
    _ = load_checkpoint_torch_compile(checkpoint, model, optimizer)
    model.to(config["device"])
    tokenizer = Tokenizer.from_files(config["vocab_filepath"], config["merge_filepath"], config["special_tokens"])
    print("loaded tokenizer")
    decoder = Decoder(model, tokenizer, config["temperature"], config["p"])

    print("loaded decoder")
    # decoder = Decoder(model, tokenizer, config["temperature"], None)
    output = decoder.forward(config["prompt"], config["context_length"], config[ "max_generated_tokens"])
    print(output)

@app.local_entrypoint()
def modal_main(config="configs/base_config.json"):
    with open(config, "r") as f:
        config_dict=json.load(f)
    infer.remote(config_dict)