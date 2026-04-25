import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.optim as optim 
from typing import Optional
from collections.abc import Callable, Iterable
from math import cos 
import math
import numpy as np
from cs336_basics.transformer import softmax
def cross_entropy_loss(logits, targets):
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits[torch.arange(len(targets),device=targets.device), targets] - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=False))
    return -logits.mean()

def perplexity(ce_loss):
    return torch.exp(ce_loss)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(optim.Optimizer):
    def __init__(self, params, betas, eps, weight_decay, lr=1e-3, ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta": betas, "eps": eps, "lambda":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            lamb = group["lambda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if (len(state) == 0):
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                grad = p.grad.data
                t = state.get("t", 1)
                alpha_t = lr*((1 - beta[1]**t)**0.5)/(1 - beta[0]**t)
                p.data = p.data - lr *lamb*p.data
                state["m"] = beta[0]*state["m"] + (1- beta[0])*grad
                state["v"] = beta[1]*state["v"] + (1- beta[1])*grad**2
                p.data -= alpha_t*state["m"]/(state["v"]**0.5 + eps)
                state["t"] = t + 1 
        return loss
                
def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if (t < T_w):
        return t/T_w*alpha_max
    if (t > T_c):
        return alpha_min
    temp = (t - T_w)/(T_c - T_w)*math.pi
    return alpha_min + 0.5*(1 + cos(temp))*(alpha_max - alpha_min)

def gradient_clipping(parameters,max_l2_norm):
    temp = sum(torch.sum(param.grad**2) for param in parameters if param.grad is not None )
    norm = torch.sqrt(temp)
    if (norm >= max_l2_norm):
        factor = max_l2_norm/(norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= factor

def data_loading(x, batch_size, context_length, device):
    idx = np.random.randint(0, len(x) - context_length, size=batch_size)
    sequences = np.stack([x[i:i+context_length] for i in idx])
    targets = np.stack([x[i+1:i+context_length+1] for i in idx])
    return (torch.tensor(sequences, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device))
            
def save_checkpoint(model, optimizer, iteration, out):
    my_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(my_dict, out)

def load_checkpoint_torch_compile(src, model, optimizer):
    state_dict = torch.load(src)
    model_state = state_dict["model"]
    new_state_dict = {}
    for k, v in model_state.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]

def load_checkpoint(src, model, optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]

class Decoder():
    def __init__(self, model, tokenizer, temperature, p):
        self.model = model
        self.tokenizer = tokenizer
        self.temp = temperature
        self.p = p
    
    def top_p(self, probs):
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        running_sum = torch.cumsum(sorted_probs, dim=0)
        is_sum_greater_than_p = running_sum >= self.p
        first_idx = torch.argmax(is_sum_greater_than_p.int())
        sorted_probs = sorted_probs[:first_idx +1]
        return sorted_probs / sorted_probs.sum(), sorted_idx[:first_idx +1]

    def forward(self, prompt, context_length, max_tokens):
        print("decoder", context_length)
        encoded_prompt = self.tokenizer.encode(prompt)
        if len(encoded_prompt) > context_length:
            print("error: input is too long")
            return 
        num_tokens_generated = 0
        while True:
            sequence = torch.tensor(encoded_prompt[-context_length:])
            # print(sequence.shape, "forward loop decoder")
            sequence = sequence.unsqueeze(0)
            logits = self.model(sequence)
            if self.temp > 0:
                logits = logits/ self.temp
                probs = softmax(logits, -1)
                if (self.p is not None):
                    sorted_probs, sorted_idx = self.top_p(probs[0, -1])
                    next_token_index = torch.multinomial(sorted_probs, 1).item()
                    next_token_index = sorted_idx[next_token_index].item()
                else:
                    next_token_index = torch.multinomial(probs[0, -1], 1).item()
            else:
                probs = softmax(logits, -1)
                next_token_index = probs[0, -1].argmax().item()
            encoded_prompt.append(next_token_index)
            num_tokens_generated +=1
            if (self.tokenizer.decode_token(next_token_index) == "<|endoftext|>") or (num_tokens_generated==max_tokens):
                return self.tokenizer.decode(encoded_prompt)
            
            









                
