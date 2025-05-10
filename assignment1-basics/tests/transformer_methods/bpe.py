# https://github.com/Christine8888/cs336-assignment1-basics/blob/main/cs336_basics/old_files/bpe_passing.py

import regex as re
from collections import defaultdict
from multiprocessing import Pool
import json
import time
import joblib

N_BYTES = 256
MULTI = 8

class BPE():
    def process_document(self, text):
        # documents split by <|endoftext|> tokens
        # assume all data structures are initialized properly
        # do rough tokenization
        words = self.pattern.finditer(text)

        local_counts = defaultdict(int)
        local_words = defaultdict(bytes)
        
        # construct mapping to counts and mapping to bytes
        word_list = [match.group() for match in words]
        for word in word_list:
            local_counts[word] += 1

            if word not in local_words.keys():
                local_words[word] = self.encode(word)
        
        return local_counts, local_words

    def process_vocab(self, word):
        local_pairs = defaultdict(int)
        word_bytes = self.words[word]
        word_count = self.counts[word]

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            
            # account for all instances of the word
            local_pairs[pair] += word_count
        
        return local_pairs
        

    def __init__multi__(self, input_path: str, special_tokens: list[str] = None):
        # pre-compile regex
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.counts = defaultdict(int)
        self.pairs = defaultdict(int)
        self.words = defaultdict(bytes)
        self.merges = []
        
        with open(input_path, 'r') as file:
           all_text = file.read()
        
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        self.size = N_BYTES + len(special_tokens)
        self.sorted_vocabulary = sorted(self.vocabulary.items(), key = lambda x: len(x[1]), reverse = True)

        # split on every Nth EOT
        documents_split = all_text.split("<|endoftext|>")
        num_documents = len(documents_split)
        n_batches = MULTI
        documents_batched = [documents_split[i::n_batches] for i in range(n_batches)]
        documents_batched = [''.join(batch) for batch in documents_batched]
        
        # parallelize further steps
        with Pool(MULTI) as p:
            print(f"Processing {len(documents_batched)} documents")
            results = p.map(self.process_document, documents_batched)
        
        for local_counts, local_words in results:
            for word, count in local_counts.items():
                self.counts[word] += count
            
            for word, encoding in local_words.items():
                if word not in self.words:
                    self.words[word] = encoding
        
        with Pool(MULTI) as p:
            all_words = list(self.counts.keys())
            results = p.map(self.process_vocab, all_words)
        
        for local_pairs in results:
            for pair, count in local_pairs.items():
                self.pairs[pair] += count

    def __init__(self, input_path: str, special_tokens: list[str] = None):
        # read in all text
        with open(input_path, 'r', encoding="utf-8", errors="ignore") as file:
           all_text = file.read()
        
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        
        self.size = N_BYTES + len(special_tokens)

        # do rough tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = re.compile(PAT)
        words = self.pattern.finditer(all_text)

        # construct mapping to counts and mapping to bytes
        self.counts = defaultdict(int)
        for match in words:
            word = match.group()
            self.counts[word] += 1

        # greedy, start from longest tokens and work backwards
        self.sorted_vocabulary = sorted(self.vocabulary.items(), key = lambda x: len(x[1]), reverse = True)
        self.words = {word: self.encode(word) for word in self.counts.keys()}

        # count initial byte pairs, indexed by tuple (ind1, ind2), recording location of 1st byte
        self.pairs = defaultdict(int)
        for word in self.words.keys():
            self.count_pairs(word)

        self.merges = []

    def encode(self, word: str):
        word_bytes = word.encode('utf-8')
        i = 0
        encoding = []
        while i < len(word_bytes):
            for id, token in self.sorted_vocabulary:
                n = len(token)
                
                # look for matching token
                if word_bytes[i:i + n] == token:
                    i += len(token)
                    encoding.append(id)
                    break
        
        return encoding

    def count_pairs(self, word):
        # only used at the beginning of BPE
        word_bytes = self.words[word]
        word_count = self.counts[word]

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            
            # account for all instances of the word
            self.pairs[pair] += word_count
        
        
    def decode_pair(self, pair, string = True, flattened = False):
        byte_tuple = (self.vocabulary[pair[0]], self.vocabulary[pair[1]])
        if string:
            return str((byte_tuple[0], byte_tuple[1]))

        if flattened:
            byte_tuple = b''.join(byte_tuple)
        
        return byte_tuple
    
    def update(self):
        # select best merge
        pairs_ranked = sorted(self.pairs.items(), key = lambda x: (x[1], self.decode_pair(x[0])), reverse = True)
        # breakpoint()
        merge_pair, count = pairs_ranked[0]

        # update self.vocabulary
        self.vocabulary[self.size] = self.decode_pair(merge_pair, string = False, flattened = True)
        new_id = self.size
        self.size += 1
        for word in self.words:
            word_tokens = self.words[word]
            i = 0
            
            while i < len(word_tokens) - 1:
                cur_pair = (word_tokens[i], word_tokens[i + 1])
                if cur_pair == merge_pair:
                    if i > 0:
                        new_pair = (word_tokens[i - 1], new_id)
                        old_pair = (word_tokens[i - 1], word_tokens[i])
                        self.pairs[new_pair] += self.counts[word]
                        self.pairs[old_pair] -= self.counts[word]
                    
                    if i < len(word_tokens) - 2:
                        old_pair = (word_tokens[i + 1], word_tokens[i + 2])
                        new_pair = (new_id, word_tokens[i + 2])
                        self.pairs[new_pair] += self.counts[word]
                        self.pairs[old_pair] -= self.counts[word]
                    
                    self.words[word][i:i+2] = [new_id]
                    word_tokens = self.words[word]
                    self.pairs[cur_pair] -= self.counts[word]
                else:
                    i += 1

        del self.pairs[merge_pair]
        # print(self.vocabulary)
        byte_merge = self.decode_pair(merge_pair, string = False)
        self.merges.append(byte_merge)

    def train(self, vocab_size: int):
        start_time = time.time()
        while self.size < vocab_size and self.pairs:
            self.update()
            if self.size % 10 == 0:
                pass
                #print(self.size)
        end_time = time.time()
        print(end_time - start_time)
        return self.vocabulary, self.merges

    def save_model(self, output_path):
        serializable_vocab = {}
        for token_id, token_bytes in self.vocabulary.items():
            serializable_vocab[str(token_id)] = list(token_bytes)
        
        # convert to lists
        serializable_merges = []
        for (byte1, byte2), _ in sorted(self.merges.items(), key=lambda x: x[1], reverse=True):
            serializable_merges.append([byte1, byte2])
        
        # create dict
        model_data = {
            "vocabulary": serializable_vocab,
            "merges": serializable_merges[:self.size - len(self.vocabulary)],  # Only save the top merges
            "size": self.size
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)


def train_tinystories():
    BASE_PATH = "./"
    data_path = BASE_PATH + "TinyStoriesV2-GPT4-valid.txt"
    tokenizer = BPE(data_path, special_tokens = ["<|endoftext|>"])
    vocab_size = 10000
    vocabulary, merges = tokenizer.train(vocab_size)
    joblib.dump(vocabulary, BASE_PATH + "vocab.pkl")
    joblib.dump(merges, BASE_PATH + "merges.pkl")    


if __name__=="__main__":
    train_tinystories()