import torch
import torch.nn.functional as F
from KV_cache import KVCache
from adaptiveQuantization import adaptiveQuantization
from benchmark import memoryCalculation, adaptiveQuantization_memory
from graphical_analysis import generate_plots

cache = KVCache()
dim=4000
token=100
Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)
Wv = torch.randn(dim, dim)

steps = []
memory_saved_pct = []
mse_values = []

for i in range(token):
    print("After Token-", i + 1)
    X = torch.randn(1, dim)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    cache.add(K, V)
    K_a, V_a = cache.get()
    d = Q.shape[-1]
    scores = Q @ K_a.T / (d**0.5)
    weights = F.softmax(scores, dim=-1)

    output = weights @ V_a
    #print("K_a shape=", K_a.shape)
    #print("K_a =", K_a)
    #print("Original Output=", output)
    cache.importance_update(weights)
    #print("Importance=", cache.importance)
    K_c, V_c = adaptiveQuantization(K_a, V_a, cache.importance)
    scores_c = Q @ K_c.T / (d**0.5)
    weights_c = F.softmax(scores_c, dim=-1)
    output_c = weights_c @ V_c
    #print("K_c shape=", K_c.shape)
    #print("K_c =", K_c)
    #print("Compressed Output=", output_c)
    o_mem = memoryCalculation(K_a, V_a, bits=32)

    adaptive_memory = adaptiveQuantization_memory(K_a, V_a, cache.importance)
  
    print("Original memory=", o_mem)
    print("Memory after adaptive quantization=", adaptive_memory)
    saving_pct = ((o_mem - adaptive_memory) / o_mem) * 100
    print(
        "Memory saved=",
        (o_mem - adaptive_memory),
        " in (%)=",
        saving_pct,
    )
    mse = ((output - output_c) ** 2).mean() / (output ** 2).mean()
   
    print("MSE Error=", mse.item())
    steps.append(i + 1)
    memory_saved_pct.append(saving_pct)
    mse_values.append(mse.item())
generate_plots(steps, memory_saved_pct, mse_values)