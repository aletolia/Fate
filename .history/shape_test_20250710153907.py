import torch
from model import MultiTaskModelWithPerTaskFusion
from config import MoEConfig

# 1. Initialize model and config
config = MoEConfig()
model = MultiTaskModelWithPerTaskFusion(config=config)
model.eval()  # Set to evaluation mode

# 2. Create dummy input tensors
batch_size = 2
img_features = torch.randn(batch_size, 10, 768)  # (B, L_i, D)
text_feats = torch.randn(batch_size, 1, 768)    # (B, L_t, D)

# 3. Perform a forward pass
with torch.no_grad():
    output = model(img_features=img_features, text_feats=text_feats)

# 4. Print output shapes
print("\n--- Final Output Shapes ---")
for i, logit in enumerate(output['logits']):
    print(f"Logits for task {i}: {logit.shape}")
