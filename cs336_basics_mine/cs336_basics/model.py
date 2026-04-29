import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2/(in_features + out_features))**0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = (einsum(x, x, "... d_model, ... d_model -> ...")/self.d_model + self.eps)**0.5
        result = einsum(x,self.weight, "... d_model, d_model -> ... d_model")
        result = result/ rms_a.unsqueeze(-1)
        return result.to(in_dtype)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        if (d_ff is None):
            d_ff = d_model*8/3
        d_ff = 64 * round(d_ff / 64)
        std = (2/(d_model + d_ff))**0.5
        self.w1 = Linear(d_model,d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff,d_model,dtype=dtype, device=device)
        self.w3 = Linear( d_model,d_ff, dtype=dtype, device=device)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        temp = silu_w1_x * w3_x
        return self.w2( temp) 


def silu_activation(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        if (d_ff is None):
            d_ff = d_model*8/3
        d_ff = 64 * round(d_ff / 64)
        assert d_ff == 4*d_model
        self.w1 = Linear(d_model,d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff,d_model,dtype=dtype, device=device)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        return self.w2( silu_w1_x) 

        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int,theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        print(max_seq_len, "ROPE")
        i_vec = torch.arange(start=0, end=max_seq_len, step=1,device=device)
        k_vec = torch.arange(start=1, end=d_k//2 + 1, step=1,device=device)
        theta_vec = theta**((2*k_vec - 2)/d_k)
        angle = i_vec.unsqueeze(-1)/theta_vec.unsqueeze(0)
        cos_tensor = angle.cos()
        sin_tensor = angle.sin()
        self.register_buffer("cos", cos_tensor, persistent=False)
        self.register_buffer("sin", sin_tensor, persistent=False)

    def forward(self, x: torch.Tensor, token_positions) -> torch.Tensor:
        sequence_length, d_k, original_shape = x.shape[-2], x.shape[-1], x.shape
        x=x.reshape(*x.shape[:-1], d_k//2, 2)
        x_first = x[...,0]
        x_sec = x[...,1]
        # print(token_positions.min(), token_positions.max())
        # print(self.sin.shape)
        cos_vals = self.cos[token_positions]
        sin_vals = self.sin[token_positions]
        out_first=x_first*cos_vals - x_sec*sin_vals
        out_second=x_first*sin_vals+x_sec*cos_vals
        res=torch.stack([out_first, out_second], dim=-1)
        res=res.reshape(original_shape)
        return res

        


def softmax(x, i):
    x = torch.exp(x - x.max(dim=i, keepdim=True).values)
    x = x / x.sum(dim=i, keepdim=True)
    return x
    

def scaled_dot_product_attention(Q,K,V, mask):
    logits = einsum(Q, K, " ... q d, ... k d -> ... q k ") /(Q.shape[-1]**0.5)
    if mask is not None:
        logits = torch.where(mask, logits, -torch.inf)
    scores = softmax(logits, -1)
    return scores @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads, rope_layer, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope_layer = rope_layer
    
    def forward(self, x, token_positions):
        return multihead_self_attention(self.d_model, self.num_heads, self.q_proj.weight, self.k_proj.weight,
        self.v_proj.weight, self.output_proj.weight, x, self.rope_layer, token_positions)
    

def multihead_self_attention( 
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    rope_layer, 
    token_positions
) -> Float[Tensor, " ... sequence_length d_out"]:
    Q = einsum(q_proj_weight, in_features, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k ")
    K = einsum(k_proj_weight, in_features, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k ")
    V = einsum(v_proj_weight, in_features, "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v ")
    token_pos = torch.arange(start=0, end=in_features.shape[-2], step=1, out=None, device=in_features.device)
    mask = token_pos.unsqueeze(-1) >= token_pos.unsqueeze(0)
    
    Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... sequence_length num_heads d_k", num_heads=num_heads)
    K = rearrange(K, "... sequence_length ( num_heads d_k) -> ... sequence_length num_heads d_k", num_heads=num_heads)
    V = rearrange(V, "... sequence_length ( num_heads d_v) -> ... sequence_length num_heads d_v", num_heads=num_heads)

    Q = rearrange(Q, "... sequence_length  num_heads d_k -> ... num_heads sequence_length d_k")
    K = rearrange(K, "... sequence_length  num_heads d_k -> ... num_heads sequence_length d_k")
    V = rearrange(V, "... sequence_length  num_heads d_v -> ... num_heads sequence_length d_v")
    for i in range(len(Q.shape[:-2])):
        mask = mask.unsqueeze(0)
    if rope_layer is not None:
        Q = rope_layer(Q, token_positions)
        K = rope_layer(K, token_positions)
    out = scaled_dot_product_attention(Q,K,V, mask)
    
    out = rearrange(out, "... num_heads sequence_length d_v -> ...  sequence_length num_heads d_v")
    out = rearrange(out, "...  sequence_length num_heads d_v -> ...  sequence_length (num_heads d_v)")
    result = einsum(o_proj_weight, out, "d_model d_v, ...  sequence_length d_v -> ... sequence_length d_model")
    return result

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads, d_ff, rope_layer, device=None, dtype=None):
        super().__init__()
        self.rope_layer = rope_layer
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_layer)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        norm_x1 = self.ln1(x)
        norm_x1 = self.attn(norm_x1, token_positions)
        x = x + norm_x1
        norm_x2 = self.ln2(x)
        norm_x2 = self.ffn(norm_x2)
        x = x + norm_x2
        return x

    
    
class TransformerLM(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        print("transformer", context_length)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope_layer = RotaryPositionalEmbedding(d_model // num_heads, rope_theta, context_length)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class TransformerLMNoPE(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope_layer = None
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
class TransformerBlockNoNorm(nn.Module):
    def __init__(self,d_model,num_heads, d_ff, rope_layer, device=None, dtype=None):
        super().__init__()
        self.rope_layer = rope_layer
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_layer)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        # norm_x1 = self.ln1(x)
        norm_x1 = self.attn(x, token_positions)
        x = x + norm_x1
        # norm_x2 = self.ln2(x)
        norm_x2 = self.ffn(x)
        x = x + norm_x2
        return x

class TransformerLM_NoNorm(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope_layer = RotaryPositionalEmbedding(d_model // num_heads, rope_theta, context_length)
        self.layers = nn.Sequential(*[TransformerBlockNoNorm(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        # self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        # x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class TransformerBlockPostNorm(nn.Module):
    def __init__(self,d_model,num_heads, d_ff, rope_layer, device=None, dtype=None):
        super().__init__()
        self.rope_layer = rope_layer
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_layer)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        x = x + self.attn(x, token_positions)
        x = self.ln1(x)
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x

class TransformerLMPostNorm(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope_layer = RotaryPositionalEmbedding(d_model // num_heads, rope_theta, context_length)
        self.layers = nn.Sequential(*[TransformerBlockPostNorm(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


class TransformerBlockSiLU(nn.Module):
    def __init__(self,d_model,num_heads, d_ff, rope_layer, device=None, dtype=None):
        super().__init__()
        self.rope_layer = rope_layer
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_layer)
        self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        norm_x1 = self.ln1(x)
        norm_x1 = self.attn(norm_x1, token_positions)
        x = x + norm_x1
        norm_x2 = self.ln2(x)
        norm_x2 = self.ffn(norm_x2)
        x = x + norm_x2
        return x


class TransformerLMSiLU(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope_layer = RotaryPositionalEmbedding(d_model // num_heads, rope_theta, context_length)
        self.layers = nn.Sequential(*[TransformerBlockSiLU(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


## weight tying
class Embedding_tied(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        std = embedding_dim**(-0.5)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
class TransformerLM_tying(nn.Module):
    def __init__(self,vocab_size, context_length,d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        print("transformer", context_length)
        self.token_embeddings = Embedding_tied(vocab_size, d_model)
        self.rope_layer = RotaryPositionalEmbedding(d_model // num_heads, rope_theta, context_length)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, self.rope_layer, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embeddings.weight
    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x














       
