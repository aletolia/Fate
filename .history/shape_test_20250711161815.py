import torch
import torch.optim as optim
from model import MultiTaskModelWithPerTaskFusion
from config import MoEConfig, TrainingFlags, TASK_CONFIG, NUM_TASKS
from utils import MGDALoss

# 1. Initialize model, config, optimizer, and loss functions
# Use the same default configurations as in main.py
flags = TrainingFlags(use_pcgrad=False, loss_weighting_strategy='mgda') # Set loss weighting to MGDA
model_config = MoEConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModelWithPerTaskFusion(config=model_config, num_tasks=NUM_TASKS).to(device)
model.train()  # Set to training mode

optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)
task_criteria = {name: crit.to(device) for name, crit in TASK_CONFIG["criteria"].items()}

# Initialize MGDALoss
mgda_loss_fn = MGDALoss(model, device)

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
print("--- Running a single training step with MGDA... ---")
optimizer.zero_grad()

# Forward pass
output = model(
    img_features=dummy_batch["image"],
    text_feats=dummy_batch["text_feat"],
    missing_modality=dummy_batch["missing_modality"]
)

# Loss calculation using MGDALoss
total_loss, batch_losses = mgda_loss_fn(output, dummy_batch, task_criteria)

print("--- Calculating Task Losses (via MGDA) ---")
for task_name, loss_val in batch_losses.items():
    print(f"Loss for task '{task_name}': {loss_val:.4f}")

print(f"Total Weighted Loss (MGDA): {total_loss.item():.4f}")

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
sample_grad = model.post_fusion_moe.experts[0].gate_proj.weight.grad
if sample_grad is not None:
    print(f"Gradient for a sample weight exists. Norm: {sample_grad.norm().item()}")
else:
    print("Info: Gradient for sample weight is None (expert might not have been used).")


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