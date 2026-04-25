from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.bpe import *
from cs336_basics.transformer import *
import time
from cs336_basics.training import *

import argparse
import json
import random
import wandb
import numpy as np


def main(config):
    start_time = time.time()
    run = wandb.init(
        entity="sarako-stanford-university",
        project="cs336_assign1",
        config=config
    ) 
    torch.set_float32_matmul_precision('high')
    train_data_np = np.memmap(config["train_data_path"], dtype=np.uint16, mode="r")
    val_data_np = np.memmap(config["val_data_path"],dtype=np.uint16, mode="r")
    print("loaded data")
    if config["pos_embed"]=="nope":
        print("nope")
        model = TransformerLMNoPE(config["vocab_size"], config["context_length"],config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"])
    elif config["activation"]=="silu":
        model = TransformerLMSiLU(config["vocab_size"], config["context_length"],config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"])
        print("silu")
    elif config["tie_weights"]:
        model = TransformerLM_tying(config["vocab_size"], config["context_length"],config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"])
        print("tied")
    else:
        model = TransformerLM(config["vocab_size"], config["context_length"],config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"])
    model.to(config["device"])
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), (config["beta1"], config["beta2"]), config["eps"], config["weight_decay"], config["lr"])
    model.train()
    print("device", config["device"])

    
    for step in range(1, config["total_steps"]+1):
        if time.time() - start_time > 45 * 60 - 60:
            break
        sequences, targets=data_loading(train_data_np, config["batch_size"], config["context_length"], config["device"])
        optimizer.zero_grad()
        logits = model(sequences)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        gradient_clipping(parameters=model.parameters(), max_l2_norm=config["grad_clip_max_norm"])
        lr = learning_rate_schedule(t=step, alpha_max=config["lr"], alpha_min=config["min_lr"], T_w=config["warmup_steps"], T_c=config["total_steps"] )
        for group in optimizer.param_groups:
            group["lr"]=lr
        optimizer.step()
        run.log({"train_loss": loss.item(), "wall_clock_time": time.time() - start_time},step=step)
        if(step % config["checkpoint_interval"] == 0):
            save_checkpoint(model, optimizer, step, f"{config['checkpoint_dir']}/checkpoint_step_{step}.pt")
        if ((step-1) % config["eval_interval"] == 0):
            print("step:", step, " train_loss:", loss.item() )
            model.eval()
            with torch.no_grad():
                val_sequences, val_targets=data_loading(val_data_np, config["batch_size"], config["context_length"], config["device"])
                val_logits = model(val_sequences)
                val_loss = cross_entropy_loss(val_logits, val_targets)
                run.log({"eval_loss": val_loss.item(), "wall_clock_time": time.time() - start_time},step=step)
            model.train()

    #final validation loss
    final_val_loss = 0
    model.eval()
    for step in range(100):
        with torch.no_grad():
            val_sequences, val_targets=data_loading(val_data_np, config["batch_size"], config["context_length"], config["device"])
            val_logits = model(val_sequences)
            val_loss = cross_entropy_loss(val_logits, val_targets)
            final_val_loss += val_loss.item()

    print("Final loss", final_val_loss / 100)
    run.log({"final_eval_loss": final_val_loss / 100}, step=config["total_steps"])
    run.finish()

if __name__ == "__main__":
    config={}
    main(config)
    


