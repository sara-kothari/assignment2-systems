import regex as re
import os
from typing import BinaryIO
import multiprocessing
from collections import Counter
import pickle
from collections.abc import Iterable, Iterator
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def init_vocab():
    """
    initialises a vocabulary mapping from int (token ID in the vocabulary)
    to bytes (token bytes). Maps token ID 0 to 255 to corresponding 
    ASCII chars.
    """
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    return vocab

def create_pretoken_dict(chunk, delimiter) -> dict[tuple[bytes, ...], int]: 
    """
    takes in file (BinaryIO) and chunk_boundaries. iterates through each chunk, converting it into a 
    string, and doing pretokenization using regex-based parsing. 
    """

    pretoken_dict = {}
    sub_chunks = re.split(delimiter, chunk)
    for sub_chunk in sub_chunks:
        strings = re.finditer(PAT,sub_chunk)
        for string in strings: 
            string = string.group()
            utf8_encoded = string.encode("utf-8")
            key = tuple([bytes([b]) for b in utf8_encoded])
            if key not in pretoken_dict:
                pretoken_dict[key] = 0
            pretoken_dict[key] += 1
    return pretoken_dict


def combine_pretoken_dicts(dict_list):
    """ combines pretoken dicts.
        Args: list of dictionaries
        Returns: single merged dict
    """
    raise NotImplementedError

def merge_pretoken_dict(pretoken_dict, vocab, num_iter, special_tokens):
    """
    performs the merge process num_iter times on the pretokens, 
    returning the merges and the vocab
    """
    last_token_num = 255
    merges = []
    adj_pair_counts = {}

    #maps adj pairs to the set of tuples (keys in the pretoken_dict) in which they are.
    adj_pair_to_strs = {}

    for k, v in pretoken_dict.items(): 
        adj_pairs = zip(k, k[1:])
        for adj_pair in adj_pairs:
            if adj_pair not in adj_pair_counts:
                adj_pair_counts[adj_pair] = 0
            adj_pair_counts[adj_pair] += v

            if adj_pair not in adj_pair_to_strs:
                adj_pair_to_strs[adj_pair] = set()
            adj_pair_to_strs[adj_pair].add(k)
    

    
    for cur_iter in range(num_iter):
        # if (cur_iter % 100 == 0):
        #     print(cur_iter)
        if (cur_iter%1000==0):
            with open(DATA_PATH / "owt_vocab_sec.pkl", "wb") as f:
                pickle.dump(vocab, f)
            with open(DATA_PATH / "owt_merges_sec.pkl", "wb") as f:
                pickle.dump(merges, f)
        if not adj_pair_counts:
            break
        # most_freq_adj_pair = max(adj_pair_counts, key=lambda p: (adj_pair_counts[p], p))
        most_freq_adj_pair = max((count, pair) for pair, count in adj_pair_counts.items())[1]

        #update vocab
        merged_token = most_freq_adj_pair[0] + most_freq_adj_pair[1]
        last_token_num +=1
        vocab[last_token_num] = merged_token

        #update merges
        merges.append(most_freq_adj_pair)
        
        #update pretoken_dict
        
        for string in list(adj_pair_to_strs[most_freq_adj_pair]):
            new_string = []
            i = 0
            while (i < len(string)):
                if ((i < len(string) - 1) and ((string[i], string[i+1]) == most_freq_adj_pair)):
                    new_string.append(merged_token)
                    i+=2
                else:
                    new_string.append(string[i])
                    i+=1
            new_tuple = tuple(new_string)
            count = pretoken_dict[string]
            pretoken_dict[new_tuple] = count +  pretoken_dict.get(new_tuple, 0)
            del pretoken_dict[string]

            old_pairs = zip(string, string[1:])
            for old_pair in old_pairs:
                adj_pair_to_strs[old_pair].discard(string)
                adj_pair_counts[old_pair] -= count
                if adj_pair_counts[old_pair] <= 0:
                    del adj_pair_counts[old_pair]

            new_pairs = zip(new_tuple, new_tuple[1:])
            for adj_pair in new_pairs:
                if adj_pair not in adj_pair_counts:
                    adj_pair_counts[adj_pair] = 0
                adj_pair_counts[adj_pair] += count

                if adj_pair not in adj_pair_to_strs:
                    adj_pair_to_strs[adj_pair] = set()
                adj_pair_to_strs[adj_pair].add(new_tuple)
        del adj_pair_to_strs[most_freq_adj_pair]
    for token in special_tokens:
        last_token_num +=1
        vocab[last_token_num] = token.encode("utf-8")
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        pretoken_dict = Counter()
        delimiter = "|".join([re.escape(token) for token in special_tokens])
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            chunks.append(chunk)
        pretoken_dicts = []

        with multiprocessing.Pool(processes=num_processes) as pool:
            pretoken_dicts = pool.starmap(create_pretoken_dict, zip(chunks, [delimiter for _ in range(len(chunks))]))
        for d in pretoken_dicts:
            pretoken_dict+= Counter(d)

        vocab = init_vocab()
        num_iter = vocab_size - 256 - len(special_tokens)
        vocab, merges = merge_pretoken_dict(pretoken_dict, vocab, num_iter, special_tokens)
        return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = set()
        else:
            self.special_tokens = set(special_tokens)
        self.reverse_vocab = {}
        max_token_id = max(self.vocab)
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k
        # self.delimiter = "|".join([re.escape(token) for token in self.special_tokens])
        self.delimiter = "|".join([re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)])
        self.merged_tokens = {merges[i]:(merges[i][0] + merges[i][1], i) for i in range(len(merges))}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        merges = []
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges,special_tokens )
    
    def decode_token(self,id):
        result=self.vocab[id]
        output = result.decode(encoding="utf-8",errors="replace")
        return output


    def encode(self, text: str) -> list[int]:
        # pretokenize
        sequence = []
        if self.delimiter:
            chunks = re.split(f"({self.delimiter})", text)
        else:
            chunks = [text]
        for chunk in chunks:
            if (chunk in self.special_tokens):
                sequence.append(self.reverse_vocab[chunk.encode("utf-8")])
                continue
            strings = list(re.finditer(PAT,chunk))
            pretokens = [string.group().encode("utf-8") for string in strings]
            pretokens = [[bytes([b]) for b in pretoken] for pretoken in pretokens]
            # print("strings", [string.group() for string in strings])
            for j in range(len(pretokens)):
                pretoken = pretokens[j]
                current = pretoken
                while (True):
                    to_merge_pair = None
                    best_rank = float('inf')
                    for i in range(len(current) - 1):
                        cur_p = (current[i], current[i+1])
                        if cur_p in self.merged_tokens:
                            cur_rank = self.merged_tokens[cur_p][1]
                            if cur_rank < best_rank:
                                best_rank = cur_rank
                                to_merge_pair = cur_p
                    if to_merge_pair is None:
                        break
                    merged_token = self.merged_tokens[to_merge_pair][0]
                    new_token = []
                    i = 0
                    while (i < len(current)):
                        if ((i < len(current) - 1) and ((current[i], current[i+1]) == to_merge_pair)):
                            new_token.append(merged_token)
                            i+=2
                        else:
                            new_token.append(current[i])
                            i+=1
                    current = new_token               
                for b in current:
                    sequence.append(self.reverse_vocab[b])
                    
        return sequence

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable: 
            yield from self.encode(item)
    
    def decode(self, ids: list[int]) -> str:
        result = bytes()
        for cur_id in ids:
            result += self.vocab[cur_id]
        output = result.decode(encoding="utf-8",errors="replace")
        return output



def encode_chunk_file(args):
    filepath, start, end, vocab, merges, special_tokens, output_path, i = args
    print("process num: ", i )
    def line_iterable():
        with open(filepath, "rb") as f:
            f.seek(start)
            while f.tell() < end:
                line = f.readline()
                if not line:
                    break
                yield line.decode("utf-8", errors="replace")
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    curr_chunk_arr = np.fromiter(tokenizer.encode_iterable(line_iterable()), dtype=np.uint16)
    np.save(output_path, curr_chunk_arr)