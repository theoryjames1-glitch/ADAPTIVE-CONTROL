# 🎛 Adaptive Control Scheduling (ACS)

## Theory

### Core Idea

Adaptive Control Scheduling is **not a fixed learning rate decay**.
Instead, it frames training as a **feedback-controlled system**, where **learning rate, momentum, and other hyperparameters are treated as adaptive filter coefficients**.

* **Optimizer = One-pole filter**

  $$
  h_{t+1} = \mu_t h_t + (1-\mu_t) g_t, \qquad \theta_{t+1} = \theta_t - \alpha_t h_{t+1}
  $$

  where $g_t$ is the gradient, $h_t$ is the filtered gradient, $\alpha_t$ = LR, $\mu_t$ = momentum.

* **Scheduler = Envelope generator for coefficients**
  $\alpha_t, \mu_t$ evolve over time, rising (attack) when progress is made, and falling (release) when loss stagnates or variance increases.

* **Feedback = Loss / Reward**
  The scheduler listens to signals like:

  * Δloss = change in loss trend
  * variance = stability of loss
  * Δreward = improvement in reward (RL settings)

* **Adaptive law (attack/release):**

  $$
  \alpha_{t+1} =
  \begin{cases}
  \alpha_t \cdot u, & \text{if loss improving (attack)} \\
  \alpha_t \cdot d, & \text{if loss worsening (release)}
  \end{cases}
  $$

  with $u > 1$, $d < 1$, and clamps $\alpha \in [\alpha_{\min}, \alpha_{\max}]$.

### Why It Matters

* **Signal-driven, not time-driven:** adapts to what training is doing.
* **Balances stability vs plasticity:** variance damping prevents runaway LR, resonance resets help escape plateaus.
* **General:** works with any optimizer (SGD, Adam, Adafactor).
* **DSP framing:** hyperparameters = filter coefficients under feedback envelope control.

---

## Demo: Sine Wave Regression with ACS

Here’s a toy regression where a small neural net learns $y = \sin(x)$.
We use **ACS to adapt LR dynamically** with attack/release logic.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# Adaptive Control Scheduler
# ---------------------------
class AdaptiveScheduler:
    def __init__(self, optimizer, attack=1.1, release=0.9,
                 eps=1e-6, min_lr=1e-5, max_lr=0.5):
        self.optimizer = optimizer
        self.attack = attack
        self.release = release
        self.eps = eps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.prev_loss = None
        self.lr = optimizer.param_groups[0]['lr']

    def observe(self, loss):
        if self.prev_loss is not None:
            delta = loss - self.prev_loss
            if delta < -self.eps:   # improvement → attack
                self.lr *= self.attack
            elif delta > self.eps:  # worsening → release
                self.lr *= self.release
        self.lr = max(self.min_lr, min(self.lr, self.max_lr))
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr
        self.prev_loss = loss

    def get_lr(self):
        return self.lr

# ---------------------------
# Data: y = sin(x)
# ---------------------------
torch.manual_seed(0)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)
y = torch.sin(x)

x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]

# ---------------------------
# Model
# ---------------------------
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
scheduler = AdaptiveScheduler(optimizer, attack=1.1, release=0.9)

# ---------------------------
# Training
# ---------------------------
epochs = 200
loss_hist, lr_hist = [], []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    scheduler.observe(loss.item())

    loss_hist.append(loss.item())
    lr_hist.append(scheduler.get_lr())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | LR {scheduler.get_lr():.4f}")

# ---------------------------
# Results
# ---------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_hist)
plt.title("Training Loss (MSE)")
plt.yscale("log")

plt.subplot(1,2,2)
plt.plot(lr_hist, color="orange")
plt.title("Adaptive LR (Attack/Release Envelope)")
plt.xlabel("Epoch")
plt.show()

# Test predictions
model.eval()
with torch.no_grad():
    y_hat = model(x_test)

plt.figure(figsize=(6,4))
plt.plot(x_test.numpy(), y_test.numpy(), label="True sin(x)")
plt.plot(x_test.numpy(), y_hat.numpy(), label="Prediction")
plt.legend()
plt.title("Sine Regression with Adaptive Control Scheduling")
plt.show()
```

---

## What You’ll See

* **Loss curve:** steadily decreasing as the network learns.
* **LR curve:** rising during improvements (attack), falling when loss plateaus (release).
* **Prediction plot:** network approximates sine function well.

---

✅ This demo shows that **Adaptive Control Scheduling** is a working **feedback-based alternative** to fixed schedulers — a DSP-style envelope on optimizer coefficients.


[![Video Title](https://img.youtube.com/vi/MM62wjLrgmA/0.jpg)](https://www.youtube.com/watch?v=MM62wjLrgmA)
