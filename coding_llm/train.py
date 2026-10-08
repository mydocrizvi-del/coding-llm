import os, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import load_config
from .data import TextSequenceDataset
from .model import CodingLLM

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_device(name):
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])
    device = pick_device(cfg["runtime"]["device"])
    model = CodingLLM.from_config(cfg["model"]).to(device)
    ds = TextSequenceDataset(cfg["data"]["train"], cfg["data"]["tokenizer"], cfg["data"]["seq_len"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and cfg["runtime"]["dtype"]=="fp16"))
    model.train()
    step = 0
    out = Path(cfg["runtime"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    while step < cfg["training"]["max_steps"]:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            dtype = torch.bfloat16 if cfg["runtime"]["dtype"]=="bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type=="cuda"):
                loss = model(x, y)["loss"] / cfg["training"]["grad_accum_steps"]
            scaler.scale(loss).backward()
            if (step + 1) % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["training"]["log_every"] == 0:
                print(f"step={step} loss={loss.item()*cfg['training']['grad_accum_steps']:.4f}")
            if step % cfg["training"]["save_every"] == 0:
                torch.save({"model": model.state_dict(), "config": cfg["model"], "step": step}, out/f"step-{step}.pt")
            if step >= cfg["training"]["max_steps"]: break
        if len(loader) == 0: raise RuntimeError("Training dataset is empty.")
    return model
