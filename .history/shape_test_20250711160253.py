import torch
import torch.optim as optim
import torch.nn as nn
from model import MultiTaskModelWithPerTaskFusion
from config import MoEConfig

# 1. Initialize model, config, optimizer, and loss functions
config = MoEConfig()
model = MultiTaskModelWithPerTaskFusion(config=config)
model.train()  # Set to training mode for gradients

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Assume each task is a classification problem
loss_fns = [nn.CrossEntropyLoss() for _ in range(config.num_tasks)]

# 2. Create dummy input tensors and labels
batch_size = 4
img_features = torch.randn(batch_size, 10, 768)  # (B, L_i, D)
text_feats = torch.randn(batch_size, 1, 768)    # (B, L_t, D)
# Create dummy labels for each task, assuming num_classes is different per task
labels = [torch.randint(0, config.num_labels[i], (batch_size,)) for i in range(config.num_tasks)]

# 3. Full training loop: forward pass, loss calculation, backward pass
optimizer.zero_grad()

# Forward pass
output = model(img_features=img_features, text_feats=text_feats)
logits_list = output['logits']

# Loss calculation
total_loss = 0
for i, (logits, label, loss_fn) in enumerate(zip(logits_list, labels, loss_fns)):
    task_loss = loss_fn(logits, label)
    print(f"Loss for task {i}: {task_loss.item():.4f}")
    total_loss += task_loss

print(f"\nTotal Loss: {total_loss.item():.4f}")

# Backward pass and optimization
total_loss.backward()
optimizer.step()

print("\n--- Backward pass and optimizer step completed successfully. ---")

# 4. (Optional) Verify gradients have been computed
print(f"Gradient for a sample weight: {model.experts[0].fc1.weight.grad[0][0]}")