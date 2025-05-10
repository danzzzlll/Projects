import torch
from typing import Optional
from .attention import softmax


def generate(
    model,
    tokenizer,
    prefix: str,
    temperature: float = 1.0,
    max_tokens: int = 100,
    top_p: Optional[float] = None,
    device="cpu"
) -> str:
    
    model.eval()
    model.to(device)
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device) #(bs, seq_len)
    eos_token = tokenizer.encode('<|endoftext|>')[0]

    for n in range(max_tokens):
        last_logits = model(input_ids)
        next_token_logits = last_logits[:, -1, :]
        # temperature
        next_token_logits /= temperature
        probs = softmax(next_token_logits, dim = -1)
        # top-p
        sorted_values, sorted_indices = torch.sort(probs, descending = True)
        sorted_values = sorted_values[0]
        sorted_indices = sorted_indices[0]
        cumsum = torch.cumsum(sorted_values, dim = 0)
        max_ind = torch.searchsorted(cumsum, top_p)
        selected_indices = sorted_indices[:max_ind]
        selected_values = sorted_values[:max_ind]

        if len(selected_indices) <= 1:
            sample = torch.argmax(probs, keepdims = True, dim = -1)
        else:
            sample = selected_indices[torch.multinomial(selected_values, 1)].unsqueeze(0)
        sample = sample.to(device)

        input_ids = torch.cat((input_ids, sample), dim=-1)
        prefix += tokenizer.decode(sample.item())

        if sample.item() == eos_token:
            break

    return prefix

## test
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("C:/models/SmolLm2_360m/")
# prefix = "hello world how are you"

# rope=RoPE(theta=10_000., d_k=64, max_seq_len=128)

# model = TransformerLM(
#     vocab_size=49152, context_length=128, 
#     num_layers=12, num_heads=4, d_model=256, d_ff=10 ,
#     rope=rope
# )

# generate(
#     model=model,
#     tokenizer=tokenizer,
#     prefix="hello world how are you",
#     temperature=1.,
#     max_tokens=50,
#     top_p=.9,
# )
