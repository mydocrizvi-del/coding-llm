from pathlib import Path
import torch
from torch.utils.data import Dataset

class TextSequenceDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, seq_len: int):
        from .tokenizer import load_tokenizer
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        self.ids = load_tokenizer(tokenizer_path).encode(text).ids
        self.seq_len = seq_len
        self.count = max(0, (len(self.ids) - 1) // seq_len)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.ids[start:start+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[start+1:start+self.seq_len+1], dtype=torch.long)
        return x, y
