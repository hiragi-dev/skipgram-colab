"""Colab接続確認用スクリプト: GPUが使えるかと、GPU上で行列計算が動くかを確かめる"""
import platform
import time

print("Python:", platform.python_version())

try:
    import torch
except ImportError:
    raise SystemExit("PyTorchが見つかりません")

print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("GPUが見つかりません。ランタイムのタイプをGPUにしてください")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# 埋め込みの話に合わせて「重み行列 W (d×V) × ワンホット」を試す
V, d = 10_000, 300
W = torch.randn(d, V, device=device)
word_id = 42
one_hot = torch.zeros(V, device=device)
one_hot[word_id] = 1.0
emb = W @ one_hot
print("ワンホットで取り出した列と W[:, 42] が一致:", torch.allclose(emb, W[:, word_id]))

# GPUの速さの確認: 4096×4096の行列積
a = torch.randn(4096, 4096, device=device)
torch.cuda.synchronize()
t = time.time()
for _ in range(10):
    a @ a
torch.cuda.synchronize()
print(f"4096×4096行列積 10回: {time.time() - t:.3f}秒")
print("OK: Colabで実行できています")
