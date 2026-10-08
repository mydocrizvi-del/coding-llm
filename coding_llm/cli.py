import argparse
from .train import train as train_model
from .generate import generate as generate_text

def train():
    p=argparse.ArgumentParser(); p.add_argument("config"); train_model(p.parse_args().config)

def generate():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("tokenizer"); p.add_argument("prompt")
    p.add_argument("--max-new-tokens", type=int, default=128); p.add_argument("--temperature", type=float, default=0.7)
    a=p.parse_args(); print(generate_text(a.checkpoint,a.prompt,a.tokenizer,a.max_new_tokens,a.temperature))

def eval():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("tokenizer"); p.add_argument("dataset")
    p.parse_args()
    raise SystemExit("Evaluation harness is coming in the next milestone.")
