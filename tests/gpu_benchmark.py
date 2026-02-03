"""GPU benchmark to diagnose performance issues."""
import torch
import time

print("=" * 60)
print("GPU BENCHMARK")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# 简单矩阵乘法基准测试
print("\n[1] Matrix multiplication benchmark:")
sizes = [1024, 2048, 4096]
for size in sizes:
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  Matrix {size}x{size}: {elapsed*10:.2f} ms/op")

# SAC 风格的小网络测试
print("\n[2] SAC-style MLP forward pass:")
batch_sizes = [256, 1024, 4096]
for bs in batch_sizes:
    # 模拟 SAC 的 Q 网络
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    ).cuda()

    x = torch.randn(bs, 256, device='cuda')

    # 预热
    for _ in range(10):
        y = model(x)
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(1000):
        y = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  Batch {bs}: {elapsed:.3f}s / 1000 passes = {elapsed:.3f} ms/pass")

# SAC 风格的完整训练步骤（前向+反向）
print("\n[3] SAC-style MLP forward + backward:")
for bs in batch_sizes:
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(bs, 256, device='cuda')
    target = torch.randn(bs, 1, device='cuda')

    # 预热
    for _ in range(10):
        optimizer.zero_grad()
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(1000):
        optimizer.zero_grad()
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  Batch {bs}: {elapsed:.3f}s / 1000 steps = {elapsed:.3f} ms/step")

print("\n" + "=" * 60)
print("参考值 (RTX 4090 正常性能):")
print("  Matrix 4096x4096: ~5-8 ms/op")
print("  MLP batch 1024 forward+backward: ~0.5-1 ms/step")
print("=" * 60)
