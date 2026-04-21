import torch


def quantize(x, bits=8):
    min_X = x.min()
    max_X = x.max()
    num_levels = 2**bits - 1
    scale = (max_X - min_X) / (num_levels + 1e-10)
    x_quantized = ((x - min_X) / scale).round().clamp(0, num_levels)
    return x_quantized, scale, min_X


def dequantize(x_quantized, scale, min_X):
    return x_quantized * scale + min_X


# testing
if __name__ == "__main__":
    x = torch.tensor(
        [
            [0.6252, 0.1743, -0.9215],
            [0.3229, 0.4551, 1.8433],
            [-1.0526, -0.5796, -0.0992],
        ]
    )
    x_quantized, scale, min_X = quantize(x, bits=8)
    reconstruct = dequantize(x_quantized, scale, min_X)
    print("X=", x)
    print("Quantized=", x_quantized)
    print("Quantized=", x_quantized.shape)
    print("DeQuantized=", reconstruct)
    print("DeQuantized=", reconstruct.shape)
    print("MSE:          ", ((x - reconstruct) ** 2).mean().item())
