"""Check that the bundled CLIP BPE vocabulary can initialize the tokenizer."""
from pathlib import Path

from utils.simple_tokenizer import SimpleTokenizer


vocab = Path("data/bpe_simple_vocab_16e6.txt.gz")
if not vocab.is_file():
    raise FileNotFoundError(f"Bundled tokenizer vocabulary is missing: {vocab}")

tokenizer = SimpleTokenizer()
print(f"Tokenizer ready ({len(tokenizer.encoder)} tokens) using {vocab}")
