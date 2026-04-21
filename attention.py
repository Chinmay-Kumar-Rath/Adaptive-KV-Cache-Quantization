import torch
import torch.nn.functional as Fun

X = torch.randn(3, 4)  # 3 token 4 features each
Wq = torch.randn(4, 4)
Wk = torch.randn(4, 4)
Wv = torch.randn(4, 4)

Q = X @ Wq
K = X @ Wk
V = X @ Wv
print("Q=", Q)
print("K=", K)
print("V=", V)
d = Q.shape[-1]  # If Q.shape=(3,4) then Q.shape[-1]=4
scores = Q @ K.T / (d**0.5)
weights = Fun.softmax(
    scores, dim=-1
)  # dim=-1 means we want softmax to move column wise that is one token over all others
output = weights @ V

print("Scores=", scores)
print("Weights(attention)=", weights)
print("Output=", output)
print("\nRow sums (should be 1):", weights.sum(dim=-1))
