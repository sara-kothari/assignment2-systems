import torch 
from einops import einsum
import triton
import triton.language as tl

@torch.compile
def fa_pytorch_backward(Q,K,V,O,L, dO, is_causal):
    D = torch.sum(O * dO, dim=-1)
    d = Q.shape[-1]
    S = einsum(Q, K, "... nq d, ... nk d  -> ... nq nk ")/d**0.5
    if is_causal:
        q_idx = torch.arange(start=0,end=Q.shape[1] , step=1, out=None, device=Q.device).unsqueeze(-1)
        k_idx = torch.arange(start=0,end=K.shape[1] , step=1, out=None, device=Q.device).unsqueeze(0)
        mask = k_idx > q_idx
        S = torch.where(mask, S + -1e6, S)
    P = torch.exp(S - L.unsqueeze(-1))
    dV = einsum(P, dO,"... nq nk, ... nq d -> ... nk d " )
    dP = einsum(dO, V, "... nq d, ... nk d -> ... nq nk")
    dS = P * (dP - D.unsqueeze(-1))
    dQ = einsum(dS, K, "... nq nk, ... nk d -> ... nq d ")/d**0.5
    dK = einsum(dS, Q, "... nq nk, ... nq d -> ... nk d ")/d**0.5
    return dQ, dK, dV
    
class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    
    def forward(ctx, Q, K, V, is_causal=False):
        #Q.shape = B, T, D
        Bq = 16
        Bk = 16
        d = Q.shape[-1]
        O = torch.zeros_like(Q)
        #L.shape = BxT 
        L = torch.zeros(*Q.shape[:-2],Q.shape[-2] , device=Q.device, dtype=Q.dtype)
        mask = None
        q_idx = None
        k_idx = None
        for i in range(0, Q.shape[-2], Bq):
            Q_i = Q[:, i:i + Bq, :]
            O_i = torch.zeros_like(Q_i, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(*Q.shape[:-2], Bq, device=Q.device, dtype=Q.dtype)
            m_i = torch.ones_like(l_i, device=Q.device, dtype=Q.dtype)*float("-inf")
            for j in range(0, K.shape[-2], Bk):
                
                K_j = K[:, j:j + Bk, :]
                V_j = V[:, j: j+ Bk, :]
                S_ij = einsum(Q_i, K_j, "... bq d, ... bk d -> ... bq bk")/d**0.5
                if is_causal:
                    q_idx = torch.arange(start=i,end=i + Bq , step=1, out=None, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(start=j,end=j + Bk , step=1, out=None, device=Q.device).unsqueeze(0)
                    mask = k_idx > q_idx
                    S_ij = torch.where(mask, S_ij + -1e6, S_ij)
                prev_m = m_i
                m_i = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(-1))
                l_i = torch.exp(prev_m - m_i)*l_i + torch.sum(P_ij, dim=-1)
                O_i = torch.exp(prev_m - m_i).unsqueeze(-1)*O_i + P_ij @ V_j
            O_i = (1/l_i.unsqueeze(-1))*O_i
            L_i = m_i + torch.log(l_i)
            O[:, i:i + Bq, :] = O_i
            L[:, i:i + Bq] = L_i
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.is_causal = is_causal
        return O
    
    
    @staticmethod
    def backward(ctx, dO):
        Q,K,V,O,L = ctx.saved_tensors
        dQ, dK, dV = fa_pytorch_backward(Q,K,V,O,L,dO, ctx.is_causal)
        return dQ, dK, dV, None
        

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    
    #T,B
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )
    Q_i = tl.load(Q_block_ptr)
    O_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,),float("-inf") ,dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        S_ij = tl.dot(Q_i, tl.trans(K_j))*scale
        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_idx =  tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = k_idx[None, :] > q_idx[:, None]
            S_ij = tl.where(mask, S_ij + -1e6, S_ij)
        prev_m = m_i
        m_i = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.math.exp(S_ij - m_i[:, None])
        l_i = tl.math.exp(prev_m - m_i)*l_i + tl.sum(P_ij, axis=1)
        O_i = tl.math.exp(prev_m - m_i)[:, None]*O_i + tl.dot(P_ij.to(V_j.dtype),V_j)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    O_i = (1/l_i[:, None])*O_i
    L_i = m_i + tl.math.log(l_i)
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))
    

    
class FA2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((Q.shape[0], Q.shape[1]), device=Q.device, dtype=Q.dtype)
        D = Q.shape[-1]
        scale = 1/D**0.5
        N_QUERIES = Q.shape[1]
        N_KEYS = K.shape[1]
        flash_fwd_kernel[(triton.cdiv(Q.shape[-2], Q_TILE_SIZE),Q.shape[0],)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
            )
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.is_causal = is_causal
        return O

    
    @staticmethod
    def backward(ctx, dO):
        Q,K,V,O,L = ctx.saved_tensors
        dQ, dK, dV = fa_pytorch_backward(Q,K,V,O,L,dO, ctx.is_causal)
        return dQ, dK, dV, None
    