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

    ## training tokenizer
    # vocab, merges = train_bpe(
    #     input_path=DATA_PATH / "raw_data" / "owt_train.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"]
    # )
    # with open(DATA_PATH / "owt_vocab_second.pkl", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open(DATA_PATH / "owt_merges_second.pkl", "wb") as f:
    #     pickle.dump(merges, f)
    # print("done tokenizing")


    # analysis for train_bpe_expts_owt
    with open(DATA_PATH / "tiny_stories_tokenizer_vocab.pkl", "rb") as f:
        ts = pickle.load(f)

    with open(DATA_PATH / "owt_vocab_second.pkl", "rb") as f:
        owt = pickle.load(f)
    
    ts_tok = sorted(ts.values(), key=lambda p:-len(p))[:10]
    owt_tok = sorted(owt.values(), key=lambda p:-len(p))[:10]
    print("ts", [(tok, len(tok)) for tok in ts_tok])
    print("owt", [(tok, len(tok)) for tok in owt_tok])
    print(owt_tok[0].decode("utf-8"))

   
    
    
    
    # added = d2.values() - d1.values()
    # removed = d1.values() - d2.values()

    # print("Added:", added)     # {'d'}
    # print("Removed:", removed) # {'a'}
    # changed = [k for k in d1.keys() & correct.keys() if d1[k] != correct[k]]
    # print("Changed:", changed, "d1:", [d1[i] for i in changed[:10]], "correct:", [correct[i] for i in changed[:10]]) 
    # longest = max(vocab.values(), key=len)
    # for val in vocab.values():
    #     if len(val) == len(longest):
    #         print(longest, longest.decode(encoding="utf-8",errors="replace"))
    # print(longest)
    # print(len(vocab))
    # output = longest.decode(encoding="utf-8",errors="replace")
    # print(output)
    


    ### tokenizing data stuff
    # tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    # print("tokenizer ready")
    # for split in ["train", "valid"]:
    #     filepath = DATA_PATH / "raw_data" / f"owt_{split}.txt"
    #     with open(filepath, "rb") as f:
    #         num_processes = mp.cpu_count()
    #         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    #         output_paths = [DATA_PATH / f"owt_chunk{i}_par_sec_final_rb.npy" for i in range(len(boundaries) - 1)]
    #         args = [(filepath, boundaries[i], boundaries[i+1], vocab, merges, ["<|endoftext|>"], output_paths[i], i) for i in range(len(boundaries) - 1)]
    #         print(len(boundaries))
    #         with mp.Pool(mp.cpu_count()) as pool:
    #             pool.map(encode_chunk_file, args)
    #         total_size = sum(np.load(p, mmap_mode='r').shape[0] for p in output_paths)
    #         final = np.memmap(DATA_PATH / f"owt_{split}_par_sec_final_rb.npy", dtype=np.uint16, mode='w+', shape=(total_size,))
    #         offset = 0
    #         for p in output_paths:
    #             chunk = np.load(p, mmap_mode='r')
    #             final[offset:offset + len(chunk)] = chunk
    #             offset += len(chunk)
    #             os.remove(p)
    #         print(f"saved {split}")
                    

@app.local_entrypoint()
def modal_main():
    train_and_encode_owt.remote()