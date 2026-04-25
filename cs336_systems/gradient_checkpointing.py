from torch.utils.checkpoint import checkpoint
from cs336_basics.transformer import *

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
rope_layer= RotaryPositionalEmbedding(d_model // num_heads, 10000.0, context_length))


block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)
def single_block(x):
    return block(x)

def all_blocks(x, N):
    for i in range(N):
        x = checkpoint(single_block, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(f"Total size of saved tensors in four TransformerBlocks with checkpointing:{total_size_bytes / (1024**2):.2f} MiB")


