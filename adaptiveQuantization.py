import torch
from quantize import quantize, dequantize


def adaptiveQuantization(K, V, importance):
    n = len(importance)
    k_new = [None] * n
    v_new = [None] * n
    sorted_index = sorted(range(n), key=lambda i: importance[i], reverse=True)
    for rank, i in enumerate(sorted_index):
        K1 = K[i : i + 1]
        V1 = V[i : i + 1]
        if rank < max(1, int(0.4 * n)):  # to 40% token
            k_new[i] = K1
            v_new[i] = V1
           # level = "FP32"
        elif rank < max(2, int(0.6 * n)):  # next 40%
            k_q, s, m = quantize(K1, bits=8)
            K1 = dequantize(k_q, s, m)
            v_q, v_s, v_m = quantize(V1, bits=8)
            V1 = dequantize(v_q, v_s, v_m)
            k_new[i] = K1
            v_new[i] = V1
           # level = "INT8"
        else:
            k_q, s, m = quantize(K1, bits=4)
            K1 = dequantize(k_q, s, m)
            v_q, v_s, v_m = quantize(V1, bits=4)
            V1 = dequantize(v_q, v_s, v_m)
            k_new[i] = K1
            v_new[i] = V1
          #  level = "INT4"
        '''print(
            "Token=", i, " importance=", importance[i], " rank=", rank, " level=", level
        )'''
        # Level is commented and should only be used when printing the above statement
    return torch.cat(k_new, dim=0), torch.cat(v_new, dim=0)
