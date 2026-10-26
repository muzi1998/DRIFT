# -*- coding: utf-8 -*-
"""DataLoader that flattens per-item SID codes into token sequences.

The collate function turns ``(B, L, M)`` nested code lists into flat
``(B, L * M)`` token tensors and derives the attention masks from the padding
token, so the model sees one contiguous SID token stream per user.
"""

import torch
from torch.utils.data import DataLoader


class GenRecDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn)
    
    def collate_fn(self, batch, pad_token=0):
        histories = [item['history'] for item in batch]       
        targets = [item['target'] for item in batch]          
        sim_items_raw = [item['sim_items'] for item in batch] 
        sidtier_features = [item['sidtier'] for item in batch]
        
        mode = batch[0].get('mode', 'train')

        flattened_histories = torch.tensor([
            [code for item_codes in history_seq for code in item_codes] 
            for history_seq in histories
        ], dtype=torch.int64)
        
        flattened_targets = torch.tensor(targets, dtype=torch.int64)
        
        if mode == 'train':
            flattened_sims = torch.tensor([
                [code for item_codes in seq for code in item_codes]
                for seq in sim_items_raw
            ], dtype=torch.int64)
            
        else:
            flattened_sims = torch.tensor([
                [
                    [code for item_codes in candidate_seq for code in item_codes]
                    for candidate_seq in candidates
                ]
                for candidates in sim_items_raw
            ], dtype=torch.int64)

        sidtier_tensor = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in sidtier_features], dim=0)

        attention_masks = (flattened_histories != pad_token).long()
        sim_attention_masks = (flattened_sims != pad_token).long()

        return {
            'history': flattened_histories,            
            'target': flattened_targets,               
            'attention_mask': attention_masks,         
            'sim_input_ids': flattened_sims,           
            'sim_attention_mask': sim_attention_masks, 
            'sidtier': sidtier_tensor,
            'mode': mode
        }
