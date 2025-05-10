# https://github.com/Christine8888/cs336-assignment1-basics/blob/main/cs336_basics/tokenizer.py

from typing import Iterable
import regex as re
import pickle
import random
import numpy as np
import multiprocessing
from functools import partial

MULTI = 32 #max(multiprocessing.cpu_count() - 1, 1)
CHUNK_SIZE = 10_000_000

class Tokenizer():
    """Tokenizer given a BPE vocabulary and merges."""

    def __init__(self, vocab, merges, special_tokens = None):
        """Initialize the tokenizer.
        
        vocab: vocabulary
        merges: merges
        special_tokens: special tokens
        """
        self.id_to_token = vocab
        self.token_to_id = {bytes(v): int(k) for k, v in vocab.items()}
        
        self.merges = {tuple(k): i for i, k in enumerate(merges)}
        self.size = len(vocab)
        # assume they are already in the BPE vocabulary
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        # sort special tokens by length
        self.special_tokens.sort(key=len, reverse=True)
        
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        """Initialize the tokenizer from a BPE vocabulary and merges.
        
        vocab_filepath: path to the vocabulary, pickled
        merges_filepath: path to the merges, pickled
        special_tokens: special tokens
        """

        # read in pickle files
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def split_by_special_tokens(self, text):
        """Split the text by special tokens.
        
        text: text to split
        """

        if not self.special_tokens:
            return [text]
        
        # escape tokens for regex and join them
        token_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        # split and keep delimiters
        split = re.split(f"({token_pattern})", text)
        return split

    def encode(self, text):
        """Encode a text string into a list of token IDs, accounting for special tokens.
        
        text: text to encode
        """

        chunks = self.split_by_special_tokens(text)
        word_list = []
        
        # first split by special tokens
        for piece in chunks:
            if piece in self.special_tokens:
                word_list.append(piece)
            else:
                word_list.extend(match.group() for match in re.finditer(self.pattern, piece))

        # encode words one by one
        encoded = []
        n_words = len(word_list)
        
        for i, word in enumerate(word_list):
            if word in self.special_tokens:
                # special token index
                encoded.append(self.token_to_id[word.encode('utf-8')])
            else:
                # compute merges and encode
                merged = self.encode_word_from_merges(word)
                encoded.extend([self.token_to_id[b] for b in merged])
            
            # if i % 1000000 == 0:
                # print(f"encoded {i}/{n_words} words")
        
        return encoded
    
    def encode_word_from_merges(self, word):
        """Encode a word given the merge list, always taking the first appropriate merge.
        
        word: string to encode
        """

        # encode as raw bytes
        byte_list = word.encode('utf-8')
        byte_list = [bytes([b]) for b in byte_list]

        while len(byte_list) > 1:
            first_merge = None
            first_idx = float('inf')
            first_pos = None

            # check all possible pairs to merge
            for i in range(len(byte_list) - 1):
                byte_pair = (byte_list[i], byte_list[i + 1])
                if byte_pair in self.merges:
                    if self.merges[byte_pair] < first_idx:
                        # get earliest merge (from BPE training)
                        first_merge = byte_pair[0] + byte_pair[1]
                        first_idx = self.merges[byte_pair]
                        first_pos = i

            if first_merge is None:
                # no more valid merges to make
                break
            
            # merge the first pair and update bytes
            byte_list = byte_list[:first_pos] + [first_merge] + byte_list[first_pos + 2:]
        
        return byte_list
    
    def _process_chunk(self, text_chunk):
        return self.encode(text_chunk)
    
    def encode_iterable(self, iterable: Iterable[str], one_at_a_time = True):
        """
        Reads a file in streaming fashion, chunks it by special tokens,
        and encodes each chunk using multiprocessing.
        one_at_a_time: if True, yield one token at a time; memory-efficient
        """

        batch_num = 0
        special_token = "<|endoftext|>"
        token_len = len(special_token)
        
        def generate_chunks(chunk_size = None):
            """Internal generator that reads and chunks the file"""
            if chunk_size is None:
                chunk_size = CHUNK_SIZE
            leftover = ""
            
            f = iterable
            while True:
                # Read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = ""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token)

                if last_eot_idx == -1:
                    # no complete document in this chunk
                    # keep everything in leftover for the next read
                    leftover = block
                else:
                    # up through last_eot_idx is a complete set of docs
                    # generators yield result but do not close function
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len :]

            # yield leftover text
            if leftover:
                yield leftover
        
        if one_at_a_time:
            # memory efficient, read small chunks
            chunks = generate_chunks(chunk_size = 1000)
        else:
            chunks = generate_chunks()
        
        # yield one at a time, most memory efficient
        if one_at_a_time:
            for chunk in chunks:
                for token in self.encode(chunk):
                    yield token
        
        # yield chunks at a time (up to all at once)
        else:
            all_tokens = []
            
            with multiprocessing.Pool(processes=MULTI) as pool:
                process_func = partial(self._process_chunk)
                
                while True:
                    print(f"Processing batch {batch_num}", flush=True)
                    batch_num += 1

                    # collect a batch of chunks
                    batch = []
                    for _ in range(MULTI):
                        try:
                            chunk = next(chunks)
                            batch.append(chunk)
                        except StopIteration:
                            break
                    
                    if not batch:
                        break

                    # process in batches
                    results = pool.map(process_func, batch)
                    for result in results:
                        all_tokens.extend(result)
            
            yield all_tokens
    
    def decode(self, ids):
        """Decode a list of token IDs into a string, using the BPE vocabulary.
        
        ids: list of token IDs
        """

        # first decode ids into bytes
        byte_list = b""
        for id in ids:
            if str(id) in self.id_to_token:
                byte_list += bytes(self.id_to_token[str(id)])
            elif id in self.id_to_token:
                byte_list += bytes(self.id_to_token[id])
            else:
                # use unicode replacement character
                byte_list += b"U+FFFD"
        
        # then decode bytes into text
        return byte_list.decode('utf-8', errors='replace')
    
def chunked_text_generator(filepath):
    """Old function to chunk text by special tokens."""

    with open(filepath, 'r') as f:
        buffer = []
        total_chars = 0
        for line in f:
            buffer.append(line)
            total_chars += len(line)
            if total_chars >= CHUNK_SIZE:
                yield ''.join(buffer)
                buffer = []
                total_chars = 0
        if buffer:
            yield ''.join(buffer)


def test_tokenizer(data_path = 'test.txt'):
    """Test the tokenizer and get compression ratio + longest token."""

    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath = f"vocab.pkl", merges_filepath = f"merges.pkl", 
                                     special_tokens = special_tokens)
    
    text = open(data_path, "r").read()
    text = text.split("<|endoftext|>")
    
    # set random seed for reproducibility
    # random.seed(42)
    sampled_texts = random.sample(text, 10)

    compression_ratios = []
    for text in sampled_texts:
        text_bytes = text.encode('utf-8')
        encoded_ids = tokenizer.encode(text)
        print([tokenizer.decode([id]) for id in encoded_ids])
        try:
            compression_ratios.append(len(text_bytes) / len(encoded_ids))
        except:
            pass
    
    print(f"Average compression ratio: {sum(compression_ratios) / len(compression_ratios)}")

    print('Longest token:')
    longest_token = max(tokenizer.token_to_id.keys(), key=len)
    print(bytes(longest_token).decode('utf-8'))

if __name__=="__main__":
    test_tokenizer()