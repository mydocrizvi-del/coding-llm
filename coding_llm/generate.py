import torch
from .model import CodingLLM
from .tokenizer import load_tokenizer

@torch.no_grad()
def generate(checkpoint, prompt, tokenizer_path, max_new_tokens=128, temperature=0.7, top_k=50, device="auto"):
    device = torch.device("cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device))
    ckpt = torch.load(checkpoint, map_location=device)
    model = CodingLLM.from_config(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tok = load_tokenizer(tokenizer_path)
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)["logits"][:, -1, :]
        logits = logits / max(temperature, 1e-5)
        if top_k:
            vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
            filt = torch.full_like(logits, float("-inf")); filt.scatter_(1, idx, vals); logits = filt
        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())
