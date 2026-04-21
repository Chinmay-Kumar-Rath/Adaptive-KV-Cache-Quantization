import os
import psutil
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
from KV_cache import KVCache
from adaptiveQuantization import adaptiveQuantization
from benchmark import memoryCalculation, adaptiveQuantization_memory
from graphical_analysis import generate_plots


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2').to(device)
embeddings = model.get_input_embeddings()
    
    
dim = 768 # GPT-2 uses a 768-dimensional latent space 
    
   
sample_text = "The context window of large language models is growing, but the KV cache is a bottleneck. " * 500 # Generating a realistic sequence of text to test
input_ids = tokenizer(sample_text, return_tensors="pt")['input_ids'][0]
token = min(4500, len(input_ids)) #tesing on 4500 real tokens (its takes longer time but 400 token can be taken to complete in shorter time)
scale = 1.0 / (dim ** 0.5)
cache = KVCache()

Wq = torch.randn(dim, dim).to(device)*scale
Wk = torch.randn(dim, dim).to(device)*scale
Wv = torch.randn(dim, dim).to(device)*scale

steps = []
memory_saved_pct = []
mse_values = []

for i in range(token):
    print("After Token-", i + 1)
    
    token_id = input_ids[i].unsqueeze(0)
        
    # We use torch.no_grad() because we are simulating inference, not training
    with torch.no_grad():
        X = embeddings(token_id).to(device) # Shape will be (1, 768)

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
if torch.cuda.is_available():
    print("GPU memory allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
process = psutil.Process(os.getpid())
memory_in_mb = process.memory_info().rss / (1024 * 1024)
#print(f"CPU memory allocated: {memory_in_mb:.2f} MB")
print("CPU memory allocated:", memory_in_mb, "MB")
generate_plots(steps, memory_saved_pct, mse_values)