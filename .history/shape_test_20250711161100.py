
import torch
import torch.optim as optim
from model import MultiTaskModelWithPerTaskFusion
from config import MoEConfig, TrainingFlags, TASK_CONFIG, NUM_TASKS

# 1. Initialize model, config, optimizer, and loss functions
# Use the same default configurations as in main.py
flags = TrainingFlags(use_pcgrad=False) # Disable PCGrad for simple backward pass
model_config = MoEConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModelWithPerTaskFusion(config=model_config, num_tasks=NUM_TASKS).to(device)
model.train()  # Set to training mode

optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)
task_criteria = {name: crit.to(device) for name, crit in TASK_CONFIG["criteria"].items()}

# 2. Create a dummy batch similar to what the DataLoader provides
batch_size = 4
dummy_batch = {
    "image": torch.randn(batch_size, 10, model_config.hidden_size).to(device),
    "text_feat": torch.randn(batch_size, 1, model_config.hidden_size).to(device),
    "missing_modality": [0] * batch_size,
    # Create dummy labels for each task (binary classification)
    "label": torch.randint(0, 2, (batch_size,)).float().to(device),
    "lymph_node_label": torch.randint(0, 2, (batch_size,)).float().to(device),
    "vascular_thrombus": torch.randint(0, 2, (batch_size,)).float().to(device),
    "perineural_invasion": torch.randint(0, 2, (batch_size,)).float().to(device),
    "file_path": [f"dummy_file_{i}.h5" for i in range(batch_size)]
}


# 3. Full training step: forward pass, loss calculation, backward pass
print("--- Running a single training step... ---")
optimizer.zero_grad()

# Forward pass
output = model(
    img_features=dummy_batch["image"],
    text_feats=dummy_batch["text_feat"],
    missing_modality=dummy_batch["missing_modality"]
)

# Loss calculation
task_losses = []
batch_losses = {}
print("--- Calculating Task Losses ---")
for i, task_name in enumerate(TASK_CONFIG["names"]):
    logits = output["logits"][i].view(-1)
    labels = dummy_batch[TASK_CONFIG["label_keys"][i]].float().view(-1)
    mask = labels >= 0  # In case of ignored labels (-1)

    if mask.any():
        loss_i = task_criteria[task_name](logits[mask], labels[mask])
        task_losses.append(loss_i)
        batch_losses[f"{task_name}_loss"] = loss_i.item()
        print(f"Loss for task '{task_name}': {loss_i.item():.4f}")
    else:
        task_losses.append(torch.tensor(0.0, device=device))
        batch_losses[f"{task_name}_loss"] = 0.0

# Default to simple sum of losses, as in the non-DWA/MGDA case
total_loss = torch.stack(task_losses).sum()
print(f"Total Loss: {total_loss.item():.4f}")

# Add auxiliary loss if it exists
if 'aux_loss' in output and output['aux_loss'] is not None:
    aux_loss = output['aux_loss'] * model_config.aux_loss_alpha
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")
    total_loss += aux_loss


# Backward pass and optimization
total_loss.backward()
optimizer.step()

print("--- Backward pass and optimizer step completed successfully. ---")

# 4. Verify gradients have been computed
# Check a parameter in the first expert of the MoE layer
sample_grad = model.moe.experts[0].w1.weight.grad
if sample_grad is not None:
    print(f"Gradient for a sample weight exists. Norm: {sample_grad.norm().item()}")
else:
    print("Error: Gradient for sample weight is None.")


# 5. Verify model output shapes
print("--- Verifying output shapes ---")
print(f"Logits list length: {len(output['logits'])}")
for i, logits in enumerate(output['logits']):
    # Expected: (B, 1) -> squeezed to (B,)
    print(f"Shape of logits for task {i}: {logits.shape}")

if 'aux_loss' in output and output['aux_loss'] is not None:
    print(f"Shape of auxiliary loss: {output['aux_loss'].shape}")
    print(f"Value of auxiliary loss: {output['aux_loss'].item():.4f}")

print("--- Shape & backward test completed. ---")
