from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorForInstructLM:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        length_list = [len(x['input_ids']) for x in features]
        max_length = max(length_list)
        all_input_ids = []
        all_labels = []
        all_attention_masks = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['input_ids'])
            difference = max_length - curr_length
            # left padding
            attention_mask = [0] * difference + [1] * curr_length
            pad_input_ids = [self.tokenizer.pad_token_id] * difference + features[idx]['input_ids']
            labels = [-100] * difference + features[idx]['labels']

            all_input_ids.append(pad_input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        all_input_ids = torch.LongTensor(all_input_ids)
        all_labels = torch.LongTensor(all_labels)
        all_attention_masks = torch.LongTensor(all_attention_masks)

        batch = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }
        return batch

