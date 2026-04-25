T = 16384
D = 1600
V = 50257
D_ff = 4288
L =48


result = 2*T*D*V + L*(8*T*D**2 + 4*T**2*D + 6*T*D*D_ff)
print(result)

swiglu = L*6*T*D*D_ff
attn_qkvo = L*(8*T*D**2)
attn_scores = L*4*T**2*D
embedding = 2*T*D*V

print("swiglu", swiglu, swiglu/result)
print("attn_qkvo", attn_qkvo, attn_qkvo/result)
print("attn_scores",attn_scores,attn_scores/result  )
print("embedding", embedding, embedding/result)
