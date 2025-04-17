
from transformers import DataCollatorWithPadding
import torch

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        batch_prompts, targets = zip(*batch)
            
        flat_prompts = []
        for prompts in batch_prompts:

            for i in range(len(prompts["input_ids"])):

                flat_prompts.append({k: prompts[k][i] for k in prompts})
        
        padded_flat = self.data_collator(flat_prompts)
        
        num_prompts = len(batch_prompts[0]["input_ids"])
        batch_size = len(batch_prompts)

        reconstructed = {}
        for key, tensor in padded_flat.items():
            reconstructed[key] = tensor.view(batch_size, num_prompts, -1)
        
        targets = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else targets
        
        return reconstructed, targets