import json
import time

from cs336_basics.bpe import train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--vocab_size", type=int, required=True)
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()
import pickle


def main():
    start_time = time.time()
    vocab, merges  = train_bpe(
        input_path=args.dataset,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)
    with open(f"{args.name}_tokenizer_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{args.name}_tokenizer_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    

if __name__ == "__main__":
    main()
    
