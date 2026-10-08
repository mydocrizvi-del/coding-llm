import math
import torch

@torch.no_grad()
def evaluate(model, dataset, batch_size=1, device=None):
    device = device or next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = model(x, y)["loss"]
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    if not total_tokens:
        raise RuntimeError("Evaluation dataset is empty.")
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(min(loss, 20.0))}
