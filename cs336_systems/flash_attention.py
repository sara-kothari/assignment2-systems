import torch 
from einops import einsum
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl= None



def flatten(x):
    if x.dim() == 4:
        B, H, T, D = x.shape
        return x.reshape(B * H, T, D)
    return x

@torch.compile
def fa_pytorch_backward(Q,K,V,O,L, dO, is_causal):
    dO = flatten(dO)
    D = torch.sum(O * dO, dim=-1)
    d = Q.shape[-1]
    S = einsum(Q, K, "... nq d, ... nk d  -> ... nq nk ")/d**0.5
    N_Q = S.shape[-2]
    N_K = S.shape[-1]
    if is_causal:
        # q_idx = torch.arange(start=0,end=Q.shape[1] , step=1, out=None, device=Q.device).unsqueeze(-1)
        # k_idx = torch.arange(start=0,end=K.shape[1] , step=1, out=None, device=Q.device).unsqueeze(0)
        # mask = k_idx > q_idx
        mask = torch.triu(torch.ones((N_Q, N_K), device=Q.device, dtype=torch.bool),diagonal=1)[None, :, :]
        S = torch.where(mask, S + -1e6, S)
    
    # P = torch.exp(S - L.unsqueeze(-1))
    P = torch.exp(S -L[..., :, None])
    dV = einsum(P, dO,"... nq nk, ... nq d -> ... nk d " )
    dP = einsum(dO, V, "... nq d, ... nk d -> ... nq nk")
    dS = P * (dP - D[..., :, None])
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
    # m_i = tl.full((Q_TILE_SIZE,),float("-inf") ,dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -1.0e9, dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        # S_ij = tl.dot(Q_i, tl.trans(K_j))*scale
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_idx =  tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = k_idx[None, :] > q_idx[:, None]
            S_ij = tl.where(mask, S_ij + -1e6, S_ij)
        prev_m = m_i
        m_i = tl.maximum(m_i, tl.max(S_ij, axis=1))
        # P_ij = tl.math.exp(S_ij - m_i[:, None])
        P_ij = tl.exp((S_ij - m_i[:, None]))
        l_i = tl.math.exp(prev_m - m_i)*l_i + tl.sum(P_ij, axis=1)
        O_i = tl.dot(P_ij.to(V_j.dtype),V_j, acc=tl.math.exp(prev_m - m_i)[:, None]*O_i )
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    O_i = (1/l_i[:, None])*O_i
    L_i = m_i + tl.math.log(l_i)
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))
    


@triton.jit
def flash_bwd_kernel_1(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
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
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
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
    Q_i  = tl.load(Q_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    l_i  = tl.load(L_block_ptr)
    D_i  = tl.load(D_block_ptr)
    dQ_i = tl.zeros((Q_TILE_SIZE, D),dtype=tl.float32)
    q_end = (query_tile_index + 1) * Q_TILE_SIZE
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_start = j * K_TILE_SIZE
        if is_causal and k_start >= q_end:
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        else:
            K_j = tl.load(K_block_ptr)
            V_j = tl.load(V_block_ptr)
            S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
            if is_causal:
                q_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
                k_idx = tl.arange(0, K_TILE_SIZE) + k_start
                mask  = k_idx[None, :] > q_idx[:, None]
                S_ij  = tl.where(mask, S_ij - 1e6, S_ij)

            P_ij  = tl.exp(S_ij - l_i[:, None])
            dP_ij = tl.dot(dO_i.to(V_j.dtype), tl.trans(V_j))
            dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
            dQ_i = tl.dot(dS_ij.to(K_j.dtype), K_j, acc=dQ_i)
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty))

    
@triton.jit
def flash_bwd_kernel_2(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_start = i * Q_TILE_SIZE
        if is_causal and q_start + Q_TILE_SIZE <= key_tile_index * K_TILE_SIZE:
            Q_block_ptr  = Q_block_ptr.advance((Q_TILE_SIZE, 0))
            dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
            L_block_ptr  = L_block_ptr.advance((Q_TILE_SIZE,))
            D_block_ptr  = D_block_ptr.advance((Q_TILE_SIZE,))
        else:
            Q_i  = tl.load(Q_block_ptr)
            dO_i = tl.load(dO_block_ptr)
            l_i  = tl.load(L_block_ptr)
            D_i  = tl.load(D_block_ptr)
            S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
            if is_causal:
                q_idx = tl.arange(0, Q_TILE_SIZE) + q_start
                k_idx = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
                mask  = k_idx[None, :] > q_idx[:, None]
                S_ij  = tl.where(mask, S_ij - 1e6, S_ij)
            P_ij  = tl.exp(S_ij - l_i[:, None])
            dP_ij = tl.dot(dO_i.to(V_j.dtype), tl.trans(V_j))
            dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
            dV_j = tl.dot(tl.trans(P_ij).to(dO_i.dtype), dO_i, acc=dV_j)
            dK_j = tl.dot(tl.trans(dS_ij).to(Q_i.dtype), Q_i, acc=dK_j )
            Q_block_ptr  = Q_block_ptr.advance((Q_TILE_SIZE, 0))
            dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
            L_block_ptr  = L_block_ptr.advance((Q_TILE_SIZE,))
            D_block_ptr  = D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty))
    
    
class FA2Triton(torch.autograd.Function):
    print("in FA2 yay")
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        #print(f"Q, {Q.shape},{Q.dtype} K {K.shape}, {K.dtype}, V {V.shape}, {V.dtype}")
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        ctx.q_shape = Q.shape
        ctx.k_shape = K.shape
        ctx.v_shape = V.shape
        Q = flatten(Q)
        K = flatten(K)
        V = flatten(V)
        # Q = Q.to(torch.bfloat16)
        # K = K.to(torch.bfloat16)
        # V = V.to(torch.bfloat16)
        Q = Q.to(V.dtype)
        K = K.to(V.dtype)
        
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
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
        O = O.reshape(ctx.q_shape)
        return O

    
    # @staticmethod
    # def backward(ctx, dO):
    #     Q,K,V,O,L = ctx.saved_tensors
    #     dQ, dK, dV = fa_pytorch_backward( Q,K,V,O,L,dO, ctx.is_causal)
    #     dQ = dQ.reshape(ctx.q_shape)
    #     dK = dK.reshape(ctx.k_shape)
    #     dV = dV.reshape(ctx.v_shape)
    #     return dQ, dK, dV, None
    @staticmethod
    def backward(ctx, dO):
        Q1_TILE_SIZE = 64
        K1_TILE_SIZE = 64
        Q,K,V,O,L = ctx.saved_tensors
        
        dO = flatten(dO)
        dO = dO.contiguous()
        #print(f"Q, {Q.shape},{Q.dtype} K {K.shape}, {K.dtype}, V {V.shape}, {V.dtype}, O, {O.shape}, {O.dtype}, dO after flatten {dO.shape}, {dO.dtype}")
        dQ = torch.empty_like(Q, device=Q.device).contiguous()
        dK = torch.empty_like(K, device=K.device).contiguous()
        dV = torch.empty_like(V, device=V.device).contiguous()
        D = torch.sum(dO*O, dim=-1)
        d = Q.shape[-1]
        scale = 1/d**0.5
        N_QUERIES = Q.shape[1]
        N_KEYS = K.shape[1]
        
        
        flash_bwd_kernel_1[(triton.cdiv(Q.shape[-2], Q1_TILE_SIZE),Q.shape[0],)](
            Q, K, V,
            D, dO,L,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q1_TILE_SIZE,
            K1_TILE_SIZE,
            is_causal=ctx.is_causal,
            )
        Q2_TILE_SIZE = 64
        K2_TILE_SIZE = 64
        
        flash_bwd_kernel_2[(triton.cdiv(K.shape[-2], K2_TILE_SIZE),K.shape[0],)](
            Q, K, V,
            D, dO,L,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q2_TILE_SIZE,
            K2_TILE_SIZE,
            is_causal=ctx.is_causal,
            )
        
        
        dQ = dQ.reshape(ctx.q_shape)
        dK = dK.reshape(ctx.k_shape)
        dV = dV.reshape(ctx.v_shape)
        return dQ, dK, dV, None
        
        
