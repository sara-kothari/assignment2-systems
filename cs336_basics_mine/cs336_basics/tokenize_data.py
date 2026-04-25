import json
import time

from cs336_basics.bpe import *
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_filepath", type=str, required=True)
parser.add_argument("--merge_filepath", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()
import time

tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merge_filepath, ["<|endoftext|>"])
start = time.time()
print("created tokenizer")
with open(args.data, "r") as f:
    token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
end = time.time()
print(end - start)
print("encoded token_ids")
np.save(f"{args.name}_valid.npy", token_ids)
print("saved")