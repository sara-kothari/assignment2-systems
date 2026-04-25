import numpy as np
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import pickle
from cs336_basics.bpe import *
import multiprocessing as mp
import time

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    memory=102400,
    timeout=43200,
    nonpreemptible=True
)
def get_compression_ratio(files, tokenizer):
    text = "<|endoftext|>".join(files)
    start_time = time.time()
    encoded_text = tokenizer.encode(text)
    total_time = time.time() - start_time
    num_tokens = len(encoded_text)
    number_bytes = len(text.encode("utf-8"))
    return number_bytes/num_tokens, number_bytes/total_time

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    memory=102400,
    timeout=43200,
    nonpreemptible=True
)
def train_and_encode_owt():
    tiny_stories_tokenizer = Tokenizer.from_files(DATA_PATH / "tiny_stories_tokenizer_vocab.pkl", DATA_PATH / "tiny_stories_tokenizer_merges.pkl",["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files(DATA_PATH / "owt_vocab_second.pkl", DATA_PATH / "owt_merges_second.pkl",["<|endoftext|>"])
    with open(DATA_PATH / "raw_data" / "owt_valid.txt", "r") as f:
        text_1 = f.read()
    owt_data_files = text_1.split("<|endoftext|>")[:10]

    with open(DATA_PATH / "raw_data" / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
        text_2 = f.read()
    tiny_stories_data_files = text_2.split("<|endoftext|>")[:10]

    ts_compression_ratio, ts_throughput = get_compression_ratio.remote(tiny_stories_data_files ,tiny_stories_tokenizer)
    owt_compression_ratio, owt_thru = get_compression_ratio.remote(owt_data_files ,owt_tokenizer)
    mix_compression_ratio, _ = get_compression_ratio.remote(owt_data_files ,tiny_stories_tokenizer)
    print("ts_compression_ratio: ", ts_compression_ratio)
    print("owt_compression_ratio: ", owt_compression_ratio)
    print("ts_thru ", ts_throughput)
    print("owt_thru: ", owt_thru)
    print("mix ration: ", mix_compression_ratio)

@app.local_entrypoint()
def modal_main():
    train_and_encode_owt.remote()