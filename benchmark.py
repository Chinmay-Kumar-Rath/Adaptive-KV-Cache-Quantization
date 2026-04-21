def memoryCalculation(K, V, bits=32):
    total_elements = K.numel() + V.numel()
    memoryInBytes = total_elements * (bits / 8)
    return memoryInBytes


def adaptiveQuantization_memory(K, V, importance):
    n = len(importance)
    total_memory = 0
    sorted_index = sorted(range(n), key=lambda i: importance[i], reverse=True)
    for rank, i in enumerate(sorted_index):
        elements = K[i : i + 1].numel() + V[i : i + 1].numel()
        if rank < max(1, int(0.4 * n)):  # to 40% token
            bits = 32
        elif rank < max(2, int(0.6 * n)):  # next 40%
            bits = 8
        else:
            bits = 4
        total_memory += elements * (bits / 8)
    return total_memory


def cal_mse(o1, o2):
    return ((o1 - o2) ** 2).mean().item()
