import argparse
from coding_llm.tokenizer import train_tokenizer

p=argparse.ArgumentParser()
p.add_argument("inputs", nargs="+")
p.add_argument("--output", default="artifacts/tokenizer.json")
p.add_argument("--vocab-size", type=int, default=32000)
a=p.parse_args()
train_tokenizer(a.inputs, a.output, a.vocab_size)
