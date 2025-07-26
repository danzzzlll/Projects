import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import time
import json
import os

from tests.transformer_methods.transformer import TransformerLM
from tests.transformer_methods.rope import RoPE
from tests.transformer_methods.utils_transformer import *
from tests.transformer_methods.optimizers import *

class TransformerTrainer:
    def __init__(self, transformer_params, optimizer_params, training_params, rope_params, load_from):
        
        self.transformer_params = transformer_params
        self.optimizer_params = optimizer_params
        self.training_params = training_params
        self.load_from = load_from

        self.checkpoint_dir = "./checkpoints/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_path = os.path.join("DATA_DIR", f"{training_params['dataset']}_tokenized-train.npy")
        self.valid_path = os.path.join("DATA_DIR", f"{training_params['dataset']}_tokenized-valid.npy")

        rope = RoPE(
            **rope_params, 
            device=self.training_params['device'],
            dtype=self.training_params['dtype']
        )

        self.transformer_params['rope'] = rope

        self.model = TransformerLM(
            **self.transformer_params, 
            device=self.training_params['device'],
            dtype=self.training_params['dtype']
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            **self.optimizer_params,
            device=self.training_params['device'],
            dtype=self.training_params['dtype']
        )

        if self.load_from is not None:
            load_checkpoint(self.load_from, self.model, self.optimizer)
            print(f"Load model from checkpoint {load_from} path")

        self.run_id = time.strftime("%m%d_%H%M%S")
        self.load_data()

    def load_data(self):
        self.train_data = np.load(self.train_path, mmap_mode='r', allow_pickle = True).astype(np.uint16)
        self.valid_data = np.load(self.valid_path, mmap_mode='r', allow_pickle = True).astype(np.uint16)
        print('loaded data')
        assert np.max(self.valid_data) <= np.iinfo(np.uint16).max

    def train(self):
        self.start_time = time.time()

        for i in range(self.training_params["n_iter"]):

            batch, targets = batchify(
                self.train_data,
                self.transformer_params['batch_size'],
                self.training_params["seq_len"],
                self.training_params["device"],
            )

            lr = cosine_scheduler(
                i, 
                self.training_params["alpha_max"], 
                self.training_params["alpha_min"], 
                self.training_params["T_w"], 
                self.training_params["T_c"]
            )

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            logits = self.model(batch)
            loss = cross_entropy(logits, targets)

            loss.backward()

            grad_norm = gradient_clipping(self.model.parameters(), 1.0) or 0.0

            self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.total_tokens += self.training_params['batch_size'] * self.training_params['seq_len']

            if i % self.training_params["checkpoint_every"] == 0:
                save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{i}.pt")
                save_checkpoint(self.model, self.optimizer, i, save_path)
            
            # compute validation loss/perplexity
            if i % self.training_params["valid_every"] == 0:
                self.validate(i)

        with open(f"results_{self.run_id}.txt", "w") as f:
            f.write(f"Total tokens: {self.total_tokens}\n")
            f.write(f"Total time: {time.time() - self.start_time}\n")

            # compute final loss/perplexity
            valid_loss = self.validate(i)
            f.write(f"Final validation loss: {valid_loss}\n")

        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_final.pt")
        save_checkpoint(self.model, self.optimizer, i, save_path)

    
    def validate(self, step):
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0.0
            
            for _ in range(self.training_params["n_valid_batches"]):
                # compute validation loss/perplexity, using same batch size and seq len as training
                batch, targets = batchify(
                    self.valid_data, 
                    self.training_params["batch_size"], 
                    self.training_params["seq_len"], 
                    self.training_params["device"]
                )
                logits = self.model(batch)
                loss = cross_entropy(logits, targets)
                valid_loss += loss.item()
            
            valid_loss /= self.training_params["n_valid_batches"]
            perplexity = np.exp(valid_loss)
            
        print(f"valid_loss {valid_loss} and perplexity {perplexity} on step {step}")
        self.model.train()

        
