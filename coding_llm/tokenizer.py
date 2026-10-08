from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]

def train_tokenizer(input_files: list[str], output: str, vocab_size: int = 32000):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, min_frequency=2)
    tokenizer.train(input_files, trainer)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    return tokenizer

def load_tokenizer(path: str):
    return Tokenizer.from_file(path)
