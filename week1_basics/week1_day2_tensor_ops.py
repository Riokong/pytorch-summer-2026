"""
Week 1 Day 2 — Tensor ops: reshape, matmul, broadcasting
Jun 30, 2026
"""

import torch

print("=" * 50)
print("PART 1: RESHAPE / VIEW / SQUEEZE / UNSQUEEZE")
print("=" * 50)

x = torch.arange(12, dtype=torch.float32)
print("Original:", x.shape)          # [12]

a = x.view(3, 4)
print("view(3,4):", a.shape)         # [3, 4]

b = x.reshape(2, 2, 3)
print("reshape(2,2,3):", b.shape)    # [2, 2, 3]

c = a.unsqueeze(0)                   # add dim at front
print("unsqueeze(0):", c.shape)      # [1, 3, 4]

d = c.squeeze(0)                     # remove dim
print("squeeze(0):", d.shape)        # [3, 4]

# view vs reshape — key difference:
# view requires contiguous memory; reshape works even if not contiguous
# safe habit: prefer reshape unless you have a reason to use view

# --- try yourself ---
# reshape x into (6, 2), then (12, 1), then back to (12,)

print()
print("=" * 50)
print("PART 2: MATMUL")
print("=" * 50)

# 2D matrix multiply
A = torch.randn(3, 4)
B = torch.randn(4, 5)

C1 = torch.matmul(A, B)             # preferred
C2 = A @ B                          # same thing, cleaner syntax
print("A @ B:", C1.shape)           # [3, 5]

# dot product (1D)
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])
print("dot product:", torch.dot(u, v))   # 1*4 + 2*5 + 3*6 = 32.0

# batched matmul — critical for BrainLRR/attention
# shape: [batch, seq, dim] x [batch, dim, seq]
Q = torch.randn(8, 10, 64)          # batch=8, seq_len=10, d_k=64
K = torch.randn(8, 10, 64)
scores = Q @ K.transpose(-2, -1)    # [8, 10, 10] — attention score matrix
print("attention scores:", scores.shape)

# element-wise multiply (NOT matmul)
X = torch.randn(3, 4)
Y = torch.randn(3, 4)
Z = X * Y                           # element-wise, same shape
print("element-wise:", Z.shape)     # [3, 4]

# --- try yourself ---
# compute softmax of scores along last dim: torch.softmax(scores, dim=-1)
# what shape do you get?

print()
print("=" * 50)
print("PART 3: BROADCASTING")
print("=" * 50)

# Rule: dimensions are compared right-to-left
# Each dim must be equal, or one of them must be 1 (it gets expanded)

# Example 1: add scalar to matrix
M = torch.ones(3, 4)
print("M + 5:", (M + 5).shape)      # [3, 4] — scalar broadcasts to all elements

# Example 2: add vector to matrix
row = torch.tensor([1.0, 2.0, 3.0, 4.0])   # shape [4]
print("M + row:", (M + row).shape)           # [3, 4] — row broadcasts across rows

col = torch.tensor([[1.0], [2.0], [3.0]])   # shape [3, 1]
print("M + col:", (M + col).shape)           # [3, 4] — col broadcasts across cols

# Example 3: two broadcastable tensors
p = torch.randn(5, 1, 4)
q = torch.randn(1, 3, 4)
print("p + q:", (p + q).shape)      # [5, 3, 4]

# Example 4: real use case — subtract mean per feature
data = torch.randn(100, 64)         # 100 samples, 64 features
mean = data.mean(dim=0)             # [64]
centered = data - mean              # [100, 64] - [64] → broadcasting over samples
print("centered data:", centered.shape)

# Example 5: outer product via broadcasting
a = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)   # [3, 1]
b = torch.tensor([4.0, 5.0, 6.0]).unsqueeze(0)   # [1, 3]
outer = a * b                                      # [3, 3]
print("outer product:\n", outer)

# --- common mistake to watch ---
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 3)
# x1 + x2  → RuntimeError! shapes [3,4] and [4,3] are not broadcastable
# fix: x1 + x2.T  or reshape one of them

print()
print("=" * 50)
print("PART 4: USEFUL UTILITY OPS")
print("=" * 50)

t = torch.randn(4, 5)

print("shape:", t.shape)
print("dtype:", t.dtype)
print("device:", t.device)
print("transpose:", t.T.shape)                    # [5, 4]
print("transpose(-2,-1):", t.transpose(-2,-1).shape)  # same for 2D
print("sum all:", t.sum())
print("sum dim=0:", t.sum(dim=0).shape)           # [5]
print("sum dim=1:", t.sum(dim=1).shape)           # [4]
print("mean:", t.mean())
print("max:", t.max())
print("argmax:", t.argmax())

# type conversion
x_int = torch.tensor([1, 2, 3])
x_float = x_int.float()           # or .to(torch.float32)
print("float:", x_float.dtype)

# move to GPU (if available)
if torch.cuda.is_available():
    t_gpu = t.cuda()
    print("on GPU:", t_gpu.device)
else:
    print("no GPU — CPU only (fine for now)")

print()
print("All done. Key takeaways:")
print("  reshape/view: change shape without changing data")
print("  @  : matmul — works for 2D and batched (critical for attention)")
print("  broadcasting: right-align shapes, dims of 1 expand automatically")
