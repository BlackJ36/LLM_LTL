"""
Test 1: PyTorch GPU Training Capability
验证PyTorch能否在GPU上正常训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time


def test_gpu_available():
    """检查GPU是否可用"""
    print("=" * 50)
    print("GPU Availability Check")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"✓ GPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Compute capability: {props.major}.{props.minor}")

    return True


def test_simple_training():
    """简单的神经网络训练测试"""
    print("\n" + "=" * 50)
    print("Simple Neural Network Training Test")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建简单的MLP (类似maple中的网络结构)
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=256, output_dim=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # 模拟训练数据
    batch_size = 1024
    num_batches = 100

    print(f"Training {num_batches} batches with batch_size={batch_size}...")

    start_time = time.time()
    total_loss = 0

    for i in range(num_batches):
        # 随机数据
        x = torch.randn(batch_size, 64, device=device)
        y = torch.randn(batch_size, 10, device=device)

        # 前向传播
        pred = model(x)
        loss = criterion(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\n✓ Training completed in {elapsed:.2f}s")
    print(f"✓ Average loss: {total_loss / num_batches:.4f}")
    print(f"✓ Throughput: {num_batches * batch_size / elapsed:.0f} samples/sec")

    return True


def test_sac_like_training():
    """模拟SAC风格的训练 (双Q网络 + 策略网络)"""
    print("\n" + "=" * 50)
    print("SAC-style Training Test (Q-networks + Policy)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = 32
    action_dim = 8
    hidden_dim = 256

    # Q-functions (类似maple的SACTrainer)
    class QFunction(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, obs, action):
            return self.net(torch.cat([obs, action], dim=-1))

    # Policy network
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs):
            h = self.net(obs)
            mean = self.mean(h)
            log_std = self.log_std(h).clamp(-20, 2)
            std = log_std.exp()
            return mean, std

    qf1 = QFunction().to(device)
    qf2 = QFunction().to(device)
    policy = Policy().to(device)

    qf_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=3e-4
    )
    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    batch_size = 1024
    num_updates = 50

    print(f"Running {num_updates} SAC-style updates...")

    start_time = time.time()

    for i in range(num_updates):
        # 模拟replay buffer采样
        obs = torch.randn(batch_size, obs_dim, device=device)
        actions = torch.randn(batch_size, action_dim, device=device)
        rewards = torch.randn(batch_size, 1, device=device)
        next_obs = torch.randn(batch_size, obs_dim, device=device)

        # Q-function update
        with torch.no_grad():
            next_mean, next_std = policy(next_obs)
            next_actions = next_mean + next_std * torch.randn_like(next_std)
            target_q = rewards + 0.99 * torch.min(
                qf1(next_obs, next_actions),
                qf2(next_obs, next_actions)
            )

        q1_pred = qf1(obs, actions)
        q2_pred = qf2(obs, actions)
        qf_loss = ((q1_pred - target_q)**2).mean() + ((q2_pred - target_q)**2).mean()

        qf_optimizer.zero_grad()
        qf_loss.backward()
        qf_optimizer.step()

        # Policy update
        mean, std = policy(obs)
        new_actions = mean + std * torch.randn_like(std)
        q_new = torch.min(qf1(obs, new_actions), qf2(obs, new_actions))
        policy_loss = -q_new.mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"  Update {i+1}/{num_updates}, QF Loss: {qf_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\n✓ SAC-style training completed in {elapsed:.2f}s")
    print(f"✓ Updates per second: {num_updates / elapsed:.1f}")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PyTorch GPU Training Test Suite")
    print("=" * 60)

    all_passed = True

    if not test_gpu_available():
        print("\n⚠️ GPU not available, tests will run on CPU")

    try:
        test_simple_training()
    except Exception as e:
        print(f"\n❌ Simple training test failed: {e}")
        all_passed = False

    try:
        test_sac_like_training()
    except Exception as e:
        print(f"\n❌ SAC-style training test failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ All GPU training tests PASSED!")
    else:
        print("  ❌ Some tests FAILED")
    print("=" * 60)
