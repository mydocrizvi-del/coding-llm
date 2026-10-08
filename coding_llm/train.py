import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .data import TextSequenceDataset
from .model import CodingLLM

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_device(name):
    return torch.device(name) if name != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    device = pick_device(cfg["runtime"]["device"])
    model = CodingLLM.from_config(cfg["model"]).to(device)
    ds = TextSequenceDataset(cfg["data"]["train"], cfg["data"]["tokenizer"], cfg["data"]["seq_len"])
    if not len(ds): raise RuntimeError("Training dataset is empty or shorter than one sequence.")
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    if not len(loader): raise RuntimeError("Batch size is larger than the available dataset.")
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=cfg["training"]["weight_decay"])
    base_lr = cfg["training"]["learning_rate"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    use_amp = device.type == "cuda" and cfg["runtime"]["dtype"] in {"fp16", "bf16"}
    amp_dtype = torch.float16 if cfg["runtime"]["dtype"] == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    opt.zero_grad(set_to_none=True)
    model.train()
    optimizer_step = 0
    micro_step = 0
    out = Path(cfg["runtime"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    while optimizer_step < cfg["training"]["max_steps"]:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            micro_step += 1
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(x, y)["loss"]
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if micro_step % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step <= cfg["training"]["warmup_steps"]:
                    lr = base_lr * optimizer_step / max(1, cfg["training"]["warmup_steps"])
                    for group in opt.param_groups: group["lr"] = lr
                if optimizer_step % cfg["training"]["log_every"] == 0:
                    print(f"step={optimizer_step} loss={loss.item():.4f} ppl={torch.exp(loss.detach()).item():.2f}")
                if optimizer_step % cfg["training"]["save_every"] == 0:
                    torch.save({"model": model.state_dict(), "config": cfg["model"], "step": optimizer_step, "optimizer": opt.state_dict()}, out / f"step-{optimizer_step}.pt")
                if optimizer_step >= cfg["training"]["max_steps"]: break
    return model
