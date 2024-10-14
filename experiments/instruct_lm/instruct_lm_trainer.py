"""
python sanity_check_dataloader.py
"""

import datasets
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from experiments.instruct_lm.data_collator import (
    DataCollatorForInstructLM,
)
from experiments.instruct_lm.input_preprocess import (
    instruct_lm_preprocessor,
)


def get_instruct_lm_tokenizer(
    model_name_or_path,
) -> PreTrainedTokenizer:
    """
    Get the tokenizer for the instruct lm.
    Note: we already manually change the tokenizer_config.json and tokenizer.json
        so that token index 128002 is mapped to <turn_end>, a special token for
        the usage of chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if "llama3-8b" in model_name_or_path or "llama3_1-8b" in model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    else:
        raise NotImplementedError
    tokenizer.truncation_side = "left"

    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token_id is not None

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    return tokenizer

def main():
    model_name_or_path = "meta-llama/Meta-Llama-3-8B"
    tokenizer = get_instruct_lm_tokenizer(model_name_or_path)
    preprocessor = instruct_lm_preprocessor(
        tokenizer=tokenizer,
        max_len=2048,
        eot_id=128002,
        prepend_eos=False,
    )

    data_collator = DataCollatorForInstructLM(
        tokenizer=tokenizer,
    )

    dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
    dataset = dataset.map(
        preprocessor.process_daring_anteater,
        num_proc=32,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
    )

    dataset = dataset.shuffle()
    num_examples = len(dataset)
    break_points = [
        int(num_examples * 0.9),
        int(num_examples * 0.95),
        num_examples,
    ]
    train_dataset = dataset.select(np.arange(break_points[0]))

    train_dataloader = DataLoader(train_dataset, batch_size = 4, collate_fn=data_collator)
    batch = next(train_dataloader)
    print(batch.keys())


if __name__ == "__main__":
    main()
