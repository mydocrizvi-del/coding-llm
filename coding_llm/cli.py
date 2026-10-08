import argparse
import torch
from .train import train as train_model
from .generate import generate as generate_text
from .config import load_config
from .data import TextSequenceDataset
from .model import CodingLLM
from .eval import evaluate

def train():
    p=argparse.ArgumentParser(); p.add_argument("config"); train_model(p.parse_args().config)

def generate():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("tokenizer"); p.add_argument("prompt")
    p.add_argument("--max-new-tokens",type=int,default=128); p.add_argument("--temperature",type=float,default=.7)
    a=p.parse_args(); print(generate_text(a.checkpoint,a.prompt,a.tokenizer,a.max_new_tokens,a.temperature))

def eval():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("config"); p.add_argument("--batch-size",type=int,default=1)
    a=p.parse_args(); cfg=load_config(a.config)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(a.checkpoint,map_location=device)
    model=CodingLLM.from_config(ckpt["config"]).to(device); model.load_state_dict(ckpt["model"])
    ds=TextSequenceDataset(cfg["data"]["eval"],cfg["data"]["tokenizer"],cfg["data"]["seq_len"])
    print(evaluate(model,ds,a.batch_size,device))
