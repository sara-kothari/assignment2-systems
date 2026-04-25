import numpy as np
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import pickle
from cs336_basics.bpe import *
import multiprocessing as mp

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    memory=102400,
    timeout=43200,
    nonpreemptible=True
)
def train_and_encode_owt():

    with open(DATA_PATH / "tiny_stories_tokenizer_vocab.pkl", "rb") as f:
        ts = pickle.load(f)

    with open(DATA_PATH / "owt_vocab_second.pkl", "rb") as f:
        owt = pickle.load(f)
    
    ts_tok = sorted(ts.values(), key=lambda p:-len(p))[:10]
    owt_tok = sorted(owt.values(), key=lambda p:-len(p))[:10]
    print("ts", [(tok, len(tok)) for tok in ts_tok])
    print("owt", [(tok, len(tok)) for tok in owt_tok])
    print(owt_tok[0].decode("utf-8"))


@app.local_entrypoint()
def modal_main():
    train_and_encode_owt.remote()