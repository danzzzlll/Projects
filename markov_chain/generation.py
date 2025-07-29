from collections import defaultdict
import random
from typing import Generator
from tokenizer import tokenize, textify


def slice_corpus(corpus: list[str], sample_size: int) -> list[list[str]]:
    samples = (corpus[idx : idx + sample_size] for idx, _ in enumerate(corpus))
    return [s for s in samples if len(s) == sample_size]


def collect_transitions(samples: list[list[str]]) -> dict[str, list[str]]:
    transitions = defaultdict(list)
    for sample in samples:
        state = "".join(sample[0:-1])
        next_ = sample[-1]
        transitions[state].append(next_)
    return transitions


def create_chain(start_text: str, transitions: dict[str, list[str]]) -> list[str]:
    head = start_text or random.choice(list(transitions.keys()))
    return tokenize(head)


def predict_next(chain: list[str], transitions: dict[str, list[str]], sample_size: int) -> str:
    last_state = "".join(chain[-(sample_size - 1) :])
    next_words = transitions[last_state]
    return random.choice(next_words) if next_words else ""


def generate_chain(
    start_text: str, transitions: dict[str, list[str]], sample_size: int
) -> Generator[str, None, None]:
    chain = create_chain(start_text, transitions)

    while True:
        state = predict_next(chain, transitions, sample_size)
        yield state

        if state:
            chain.append(state)
        else:
            chain = chain[:-1]


def generate(source: str, start: str = "", words_count: int = 1000, sample_size: int = 4) -> str:

    corpus = tokenize(source)
    samples = slice_corpus(corpus, sample_size)
    transitions = collect_transitions(samples)

    generator = generate_chain(start, transitions, sample_size)
    chain = [next(generator) for _ in range(words_count)]
    return textify(chain)