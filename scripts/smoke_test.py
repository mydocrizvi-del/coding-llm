import torch
from coding_llm.model import CodingLLM

cfg = dict(vocab_size=1000,max_seq_len=128,d_model=128,n_layers=2,n_heads=4,d_ff=512)
m = CodingLLM.from_config(cfg)
x = torch.randint(0, 1000, (2, 32))
o = m(x, x)
print("logits:", tuple(o["logits"].shape), "loss:", float(o["loss"]))
