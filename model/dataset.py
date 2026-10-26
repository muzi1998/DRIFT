# -*- coding: utf-8 -*-
"""Dataset for SID-based generative recommendation.

Each processed parquet row holds one training/evaluation sample:

* ``history``   — list of items, each item a list of ``M`` SID codes.
* ``target``    — the ground-truth item, as ``M`` SID codes.
* ``sim_*``     — retrieved similar items used as auxiliary decoder memory
                  (``same_sid_history`` for training; ``sid_<i>_history`` columns
                  for evaluation, one candidate group per first-level code).
* ``sidtier``   — the SID-tier feature vector (first-level codebook statistics).

Raw per-level codes are offset into a single shared vocabulary via
``_apply_offset_to_code``::

    token(level i, code c) = c + i * CODEBOOK_SIZE + 1,   PAD -> 0

Sequences are left-padded so that the most recent interactions stay at the right
end, which is what the recent-window split in the method modules assumes.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def pad_or_truncate(sequence, max_len, pad_value=0):
    if not isinstance(sequence, list):
        try:
            sequence = list(sequence)
        except:
            sequence = [sequence]
            
    seq_len = len(sequence)
    if seq_len > max_len:
        return sequence[-max_len:]
    else:
        return [pad_value] * (max_len - seq_len) + sequence


def process_data(file_path, mode, max_len, sim_max_len=30, pad_token_id=0, target_depth=3):
    print(f"🔄 Loading data from: {file_path}")
    data = pd.read_parquet(file_path)
    
    processed_data = []
    pad_vec = [pad_token_id] * target_depth

    eval_sim_cols = []
    if mode == 'evaluation':
        all_cols = data.columns.tolist()
        eval_sim_cols = [c for c in all_cols if c.startswith('sid_') and c.endswith('_history')]
        eval_sim_cols.sort(key=lambda x: int(x.split('_')[1]))
        if len(eval_sim_cols) > 0:
            print(f"[Info] Evaluation mode detected. Found {len(eval_sim_cols)} sim columns.")

    for row in data.itertuples(index=False):
        history_seq = list(row.history) if row.history is not None else []
        history_processed = pad_or_truncate(history_seq, max_len, pad_value=pad_vec)

        target_code = list(row.target) if row.target is not None else pad_vec

        if mode == 'train':
            sim_data = None
            if hasattr(row, 'same_sid_history'):
                sim_data = getattr(row, 'same_sid_history')
            # Fall back to an empty list when the column is absent.
            if sim_data is None:
                sim_data = []
            
            sim_items_processed = pad_or_truncate(list(sim_data), sim_max_len, pad_value=pad_vec)
            
        else:
            sim_groups = []
            for col_name in eval_sim_cols:
                raw_seq = getattr(row, col_name, [])
                if raw_seq is None: raw_seq = []
                padded_seq = pad_or_truncate(list(raw_seq), sim_max_len, pad_value=pad_vec)
                sim_groups.append(padded_seq)
            sim_items_processed = sim_groups

        sidtier_feature = None
        if hasattr(row, 'sidtier') and row.sidtier is not None:
            sidtier_feature = np.array(row.sidtier, dtype=np.float32)
        else:
            sidtier_feature = np.zeros(256, dtype=np.float32)

        processed_data.append({
            'history': history_processed,     
            'target': target_code,           
            'sim_items': sim_items_processed, 
            'sidtier': sidtier_feature,
            'mode': mode
        })
            
    return processed_data


class GenRecDataset(Dataset):
    def __init__(self, dataset_path, mode, max_len, sim_max_len=30, 
                 codebook_size=128, target_depth=3, PAD_TOKEN=0):
        self.mode = mode
        self.CODEBOOK_SIZE = codebook_size
        self.TARGET_DEPTH = target_depth
        self.PAD_TOKEN = PAD_TOKEN
        
        self.pad_code_vec = [PAD_TOKEN] * self.TARGET_DEPTH

        self.data = process_data(
            dataset_path, mode, max_len, sim_max_len, 
            pad_token_id=PAD_TOKEN, target_depth=target_depth
        )

        for item in self.data:
            item['history'] = self._apply_offset_to_seq(item['history'])
            item['target'] = self._apply_offset_to_code(item['target'])
            
            if self.mode == 'train':
                item['sim_items'] = self._apply_offset_to_seq(item['sim_items'])
            else:
                new_groups = []
                for group in item['sim_items']:
                    new_groups.append(self._apply_offset_to_seq(group))
                item['sim_items'] = new_groups

    def _apply_offset_to_code(self, code_list):
        if not isinstance(code_list, list):
            try:
                curr_codes = code_list.tolist()
            except:
                curr_codes = list(code_list)
        else:
            curr_codes = code_list
            
        curr_codes = curr_codes[:self.TARGET_DEPTH]
        
        if all(x == self.PAD_TOKEN for x in curr_codes):
            return self.pad_code_vec
        
        return [
            int(c) + i * self.CODEBOOK_SIZE + 1 
            for i, c in enumerate(curr_codes)
        ]

    def _apply_offset_to_seq(self, seq_list):
        return [self._apply_offset_to_code(c) for c in seq_list]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    mode = batch[0]['mode']
    
    history = torch.tensor([item['history'] for item in batch], dtype=torch.long)
    target = torch.tensor([item['target'] for item in batch], dtype=torch.long)
    sim_items = torch.tensor([item['sim_items'] for item in batch], dtype=torch.long)
    sidtier = torch.tensor([item['sidtier'] for item in batch], dtype=torch.float32)
        
    return {
        'history_codes': history,
        'target_codes': target,
        'sim_item_codes': sim_items,
        'sidtier': sidtier,
        'mode': mode
    }
