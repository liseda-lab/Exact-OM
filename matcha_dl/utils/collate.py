
from transformers import DataCollatorWithPadding
import torch

class DataCollator:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        batch_prompts, targets = zip(*batch)
        batch_size = len(batch_prompts)
        G = len(batch_prompts[0]["input_ids"])
        K = len(batch_prompts[0]["input_ids"][0])

        flat_prompts = []
        for prompts in batch_prompts:
            for g in range(G):
                for k in range(K):
                    flat_prompts.append(
                        { key: prompts[key][g][k] for key in prompts }
                    )

        padded = self.data_collator(flat_prompts)

        collated = {}
        for key, tensor in padded.items():
            collated[key] = tensor.view(batch_size, G, K, -1)

        targets = torch.stack(targets) \
                  if isinstance(targets[0], torch.Tensor) \
                  else targets

        return collated, targets
