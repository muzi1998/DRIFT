"""Taobao MM fixed-K length study on audited, restored real-item identities.

Plan only by default. --check uses small synthetic models and audits real arrays;
--smoke-gpu performs two optimizer steps with the real catalog, then mini-eval;
--execute requires a matching CPU preflight report and runs the requested grid.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from hstu_paths import resolve_args, portable_path, portable_config

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import run_hstu_drift_c_v5_taobaomm as base
from hstu_fixed_k_nots import FixedKSubseqRepresentationV5NoTS
from cpu_offloaded_adamw import CPUOffloadedAdamW

HERE = Path(__file__).resolve().parent
GRID = ((100, 5), (100, 10), (500, 5), (500, 10), (500, 20),
        (1000, 5), (1000, 10), (1000, 20))
ALLOWED_GRID = GRID + ((1000, 30),)
TOPK = (1, 3, 5, 10, 20)
NAMES = ('model', 'subseq', 'huk', 'gate')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(portable_config(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sources():
    return {name: sha256(HERE/name) for name in
            ('run_hstu_drift_c_v5_taobaomm.py', 'hstu_fixed_k_nots.py',
             'cpu_offloaded_adamw.py', Path(__file__).name)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--runs', default=','.join(f'{length}:{k}' for length, k in GRID))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=20260521)
    p.add_argument('--hidden_dim', type=int, default=96)
    p.add_argument('--linear_dim', type=int, default=16)
    p.add_argument('--attention_dim', type=int, default=16)
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--use_softmax_attn', action='store_true')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--eval_item_chunk', type=int, default=16384)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--early_stop_patience', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--optimizer_state_device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--optimizer_chunk_elements', type=int, default=4194304)
    p.add_argument('--num_negatives', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--lambda_sub', type=float, default=0.05)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--cpu_threads', type=int, default=2)
    p.add_argument('--preflight_report', type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--smoke-gpu', dest='smoke_gpu', action='store_true')
    action.add_argument('--execute', action='store_true')
    a = resolve_args(p.parse_args())
    try:
        a.grid = [tuple(map(int, part.split(':'))) for part in a.runs.split(',')]
    except ValueError as exc:
        p.error(f'--runs requires comma-separated L:K pairs: {exc}')
    if len(set(a.grid)) != len(a.grid) or any(pair not in ALLOWED_GRID for pair in a.grid):
        p.error('Run grid must be a unique subset of '+str(ALLOWED_GRID))
    if a.hidden_dim % 4 or min(a.batch_size, a.eval_batch_size, a.eval_item_chunk,
                             a.epochs, a.cpu_threads, a.num_negatives,
                             a.optimizer_chunk_elements) < 1:
        p.error('Invalid positive sizes or hidden_dim not divisible by 4')
    if not 0 < a.temperature or not 0 <= a.lambda_sub:
        p.error('temperature must be positive and lambda_sub nonnegative')
    if a.preflight_report is None:
        a.preflight_report = a.output_dir/'preflight.json'
    return a


def load_input(a):
    manifest_path = a.input_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    n_items = int(manifest['num_items'])
    n_samples = int(manifest['n_samples'])
    if n_items < max(TOPK) or n_samples < 1:
        raise ValueError('Invalid catalog/sample counts')
    mapping = np.load(a.input_dir/'item_mapping.npy', mmap_mode='r', allow_pickle=False)
    if mapping.shape != (n_items,) or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError('Expected item_mapping[dense_id-1] -> original embedding row')
    if mapping.min() < 0 or len(np.unique(mapping)) != n_items:
        raise ValueError('Real-item mapping must contain unique nonnegative row IDs')
    arrays, audit = {}, {'manifest_sha256': sha256(manifest_path), 'splits': {},
                         'files_sha256': {'item_mapping.npy': sha256(a.input_dir/'item_mapping.npy')}}
    for split in ('train', 'valid', 'test'):
        items_name, targets_name = f'{split}_items.npy', f'{split}_targets.npy'
        items = np.load(a.input_dir/items_name, mmap_mode='r', allow_pickle=False)
        target = np.load(a.input_dir/targets_name, mmap_mode='r', allow_pickle=False)
        if items.shape != (n_samples, 1000) or target.shape != (n_samples,):
            raise ValueError(f'{split}: arrays do not satisfy the (N,1000)/(N,) contract')
        if items.dtype != np.dtype('int32') or target.dtype != np.dtype('int32'):
            raise ValueError(f'{split}: restored IDs must be int32')
        if target.min() < 1 or target.max() > n_items:
            raise ValueError(f'{split}: target outside fixed catalog')
        min_len, max_len, short, empty_long = 1000, 0, 0, 0
        for start in range(0, n_samples, 1024):
            block = np.asarray(items[start:start+1024])
            if block.min() < 0 or block.max() > n_items:
                raise ValueError(f'{split}: history ID outside fixed catalog')
            valid = block != 0
            if (valid[:, :-1] & ~valid[:, 1:]).any():
                raise ValueError(f'{split}: history must be contiguous and left-padded')
            lengths = valid.sum(1)
            if (lengths < 1).any():
                raise ValueError(f'{split}: empty input history')
            min_len, max_len = min(min_len, int(lengths.min())), max(max_len, int(lengths.max()))
            short += int((lengths < 50).sum())
            empty_long += int((lengths <= 50).sum())
        arrays[split] = (items, target)
        audit['splits'][split] = {'n_samples': n_samples, 'min_input_length': min_len,
                                  'max_input_length': max_len, 'short_recent_rows': short,
                                  'empty_long_rows': empty_long}
        for name in (items_name, targets_name):
            audit['files_sha256'][name] = sha256(a.input_dir/name)
    # A fixed observed real-item catalog is shared by every split and every L.
    audit.update(num_items=n_items, n_samples=n_samples,
                 protocol='fixed observed real-item catalog; exact full-catalog top20; no seen-item filter',
                 input_dir=portable_path(a.input_dir), source_hashes=sources())
    return arrays, manifest, audit


class LengthDataset(Dataset):
    def __init__(self, arrays, length):
        self.items, self.targets = arrays
        self.length = length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.items[index, -self.length:], dtype=np.int64)),
                torch.tensor(int(self.targets[index]), dtype=torch.long))


def build_modules(a, n_items, length, k, device):
    model = base.HSTUV4(n_items, 50, a.hidden_dim, a.linear_dim, a.attention_dim,
                        a.num_blocks, a.num_heads, a.dropout, a.attn_dropout,
                        use_softmax=a.use_softmax_attn).to(device)
    subseq = FixedKSubseqRepresentationV5NoTS(a.hidden_dim, K_max=k,
                                            max_long_len=length-50).to(device)
    huk = nn.ModuleList([base.HukCrossAttnV2(a.hidden_dim, num_heads=4, dropout=0.1)
                         for _ in range(a.num_blocks)]).to(device)
    gate = base.OutputGate(a.hidden_dim).to(device)
    return model, subseq, huk, gate


def forward(modules, seq):
    model, subseq, huk, gate = modules
    if seq.shape[1] <= 50:
        raise ValueError('Length experiments require L > recent_r=50')
    recent, long = seq[:, -50:], seq[:, :-50]
    recent_mask, long_mask = recent.ne(0), long.ne(0)
    has_long = long_mask.any(1)
    d0 = torch.zeros(len(seq), device=seq.device, dtype=model.item_embedding.weight.dtype)
    # One embedding operation prevents two catalog-sized gradient buffers from
    # the long/recent lookups being materialized simultaneously in backward.
    embedded = model.item_embedding(seq)*math.sqrt(model.hidden_dim)
    x_long = embedded[:, :-50]
    h_uk, valid_slots, assignment, x_long = subseq(x_long, long_mask, d0)
    position = torch.arange(50, device=seq.device)[None, :].expand(len(seq), -1)
    x = embedded[:, -50:]+model.pos_embedding(position)
    padding = ~recent_mask
    x = model.emb_dropout(x).masked_fill(padding[..., None], 0.0)
    for block, cross in zip(model.blocks, huk):
        x = block(x, padding)
        augmented = cross(x, h_uk, ~valid_slots)
        x = torch.where(has_long[:, None, None], augmented, x)
    x = model.final_norm(x).masked_fill(padding[..., None], 0.0)
    final = F.normalize(x[:, -1], p=2, dim=-1, eps=1e-6)
    augmented = gate(final, h_uk, valid_slots)
    final = torch.where(has_long[:, None], augmented, final)
    return F.normalize(final, p=2, dim=-1, eps=1e-6), assignment, x_long


def structure_loss(x_long, assignment, eps=1e-8):
    mass = assignment.sum(dim=1)
    centroid = torch.einsum('bld,blk->bkd', x_long, assignment)/mass.clamp_min(eps)[..., None]
    cosine = torch.einsum('bld,bkd->blk', F.normalize(x_long, dim=-1, eps=eps),
                           F.normalize(centroid, dim=-1, eps=eps))
    per_slot = (assignment*(1.0-cosine)).sum(1)/mass.clamp_min(eps)
    nonempty = mass > eps
    per_row = (per_slot*nonempty).sum(1)/nonempty.sum(1).clamp_min(1)
    active = nonempty.any(1)
    # Zero for all-empty batches, while retaining a valid autograd path.
    return (per_row*active).sum()/active.sum().clamp_min(1)


def sampled_loss(hidden, target_dense, output_embedding, num_negatives, temperature):
    """Original sampled-softmax RNG/logits, with a single full-weight gather.

    Using output_weight()[1:] in autograd would require an additional complete
    catalog-sized SliceBackward gradient. A single lookup for positive and
    negative IDs also avoids two catalog-sized gather backward intermediates.
    """
    temperature = max(float(temperature), 1e-8)
    n_items = output_embedding.weight.shape[0]-1
    target_index = target_dense-1
    neg_index = torch.randint(0, n_items, (len(target_dense), num_negatives), device=target_dense.device)
    collision = neg_index.eq(target_index[:, None])
    while collision.any():
        neg_index[collision] = torch.randint(0, n_items, (int(collision.sum().item()),), device=target_dense.device)
        collision = neg_index.eq(target_index[:, None])
    indices = torch.cat((target_dense[:, None], neg_index+1), dim=1)
    sampled_weight = output_embedding.weight[indices]
    pos_logits = (hidden*sampled_weight[:, 0]).sum(-1, keepdim=True)/temperature
    neg_logits = torch.einsum('bd,bnd->bn', hidden, sampled_weight[:, 1:])/temperature
    logits = torch.cat((pos_logits, neg_logits), dim=1)
    return F.cross_entropy(logits, torch.zeros(len(target_dense), dtype=torch.long, device=hidden.device))


def chunked_topk(hidden, output_weight, chunk_size, topk=20):
    """Exact full-catalog search with bounded score storage B*(chunk+topk).

    All scores use the same unnormalized output matrix as training. Output IDs
    are dense real-item identities (1..V); PAD0 is never a candidate.
    """
    topk = min(topk, len(output_weight))
    best_scores = hidden.new_empty((len(hidden), 0))
    best_ids = torch.empty((len(hidden), 0), dtype=torch.long, device=hidden.device)
    for start in range(0, len(output_weight), chunk_size):
        end = min(start+chunk_size, len(output_weight))
        scores = hidden @ output_weight[start:end].T
        count = min(topk, end-start)
        values, indices = scores.topk(count, dim=1)
        indices = indices+start+1
        merged_scores = torch.cat((best_scores, values), dim=1)
        merged_ids = torch.cat((best_ids, indices), dim=1)
        best_scores, selected = merged_scores.topk(min(topk, merged_scores.shape[1]), dim=1)
        best_ids = merged_ids.gather(1, selected)
    return best_ids


def evaluate(a, modules, arrays, length, device, limit=None):
    for module in modules:
        module.eval()
    n_samples = len(arrays[1]) if limit is None else min(limit, len(arrays[1]))
    totals = {f'{metric}@{k}': 0.0 for metric in ('Recall', 'NDCG') for k in TOPK}
    discount = 1/torch.log2(torch.arange(20, device=device, dtype=torch.float32)+2)
    with torch.inference_mode():
        for start in range(0, n_samples, a.eval_batch_size):
            end = min(start+a.eval_batch_size, n_samples)
            seq = torch.as_tensor(np.array(arrays[0][start:end, -length:], dtype=np.int64), device=device)
            target = torch.as_tensor(np.array(arrays[1][start:end], dtype=np.int64), device=device)
            h, _, _ = forward(modules, seq)
            predicted = chunked_topk(h, modules[0].output_weight(), a.eval_item_chunk)
            hit = predicted.eq(target[:, None])
            for k in TOPK:
                totals[f'Recall@{k}'] += hit[:, :k].any(1).sum().item()
                totals[f'NDCG@{k}'] += (hit[:, :k]*discount[None, :k]).sum().item()
    return {key: value/n_samples for key, value in totals.items()}


def preflight_signature(a):
    keys = ('hidden_dim', 'linear_dim', 'attention_dim', 'num_blocks', 'num_heads',
            'dropout', 'attn_dropout', 'use_softmax_attn', 'lambda_sub', 'temperature',
            'optimizer_state_device', 'optimizer_chunk_elements')
    return {key: getattr(a, key) for key in keys}


def optimizer_parity_check():
    """Compare five exact AdamW updates, including None/zero gradients."""
    torch.manual_seed(9137)
    reference = [nn.Parameter(torch.randn(11, 7)), nn.Parameter(torch.randn(23))]
    offloaded = [nn.Parameter(p.detach().clone()) for p in reference]
    standard = torch.optim.AdamW(reference, lr=1e-4, betas=(0.9, 0.98), eps=1e-8,
                                 weight_decay=0.01, foreach=False)
    cpu_state = CPUOffloadedAdamW(offloaded, lr=1e-4, betas=(0.9, 0.98), eps=1e-8,
                                  weight_decay=0.01, chunk_elements=13)
    max_diff = 0.0
    for step in range(5):
        for index, (left, right) in enumerate(zip(reference, offloaded)):
            if index == 1 and step in (0, 2):
                left.grad = right.grad = None
            else:
                grad = torch.zeros_like(left) if step == 3 else torch.randn_like(left)
                left.grad, right.grad = grad.clone(), grad.clone()
        standard.step(); cpu_state.step()
        for left, right in zip(reference, offloaded):
            max_diff = max(max_diff, float((left-right).abs().max()))
            assert torch.allclose(left, right, rtol=1e-6, atol=1e-7), 'CPU-offloaded AdamW parameter mismatch'
            if left in standard.state:
                for key in ('exp_avg', 'exp_avg_sq'):
                    expected, actual = standard.state[left][key], cpu_state.state[right][key]
                    max_diff = max(max_diff, float((expected-actual).abs().max()))
                    assert torch.allclose(expected, actual, rtol=1e-6, atol=1e-7), 'AdamW moment mismatch'
                assert int(standard.state[left]['step']) == cpu_state.state[right]['step']
    return {'passed': True, 'steps': 5, 'chunk_elements': 13, 'includes_none_gradient': True,
            'includes_zero_gradient': True, 'weight_decay': 0.01, 'max_abs_diff': max_diff}


def sampled_loss_parity_check():
    torch.manual_seed(7513)
    old_hidden = torch.randn(7, 12, requires_grad=True)
    new_hidden = old_hidden.detach().clone().requires_grad_(True)
    old_weight = torch.randn(98, 12, requires_grad=True)
    embedding = nn.Embedding(98, 12, padding_idx=0)
    embedding.weight.data.copy_(old_weight)
    target = torch.tensor([1, 7, 26, 97, 12, 29, 30])
    torch.manual_seed(4217)
    old_loss = base.sampled_softmax_loss(old_hidden, target-1, old_weight[1:], 39, 0.05)
    torch.manual_seed(4217)
    new_loss = sampled_loss(new_hidden, target, embedding, 39, 0.05)
    old_loss.backward(); new_loss.backward()
    assert torch.allclose(old_loss, new_loss, rtol=1e-6, atol=1e-6)
    assert torch.allclose(old_hidden.grad, new_hidden.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(old_weight.grad, embedding.weight.grad, rtol=1e-5, atol=1e-6)
    assert embedding.weight.grad[0].eq(0).all()
    return {'passed': True, 'same_negative_sampling_rng': True, 'single_weight_gather': True,
            'loss_abs_diff': float((old_loss-new_loss).abs()),
            'hidden_grad_max_abs_diff': float((old_hidden.grad-new_hidden.grad).abs().max()),
            'weight_grad_max_abs_diff': float((old_weight.grad-embedding.weight.grad).abs().max())}


def synthetic_checks(a):
    """Small-catalog CPU checks; never allocate the real catalog on the CPU."""
    torch.set_num_threads(a.cpu_threads)
    base.set_seed(a.seed)
    device = torch.device('cpu')
    results = []
    # Cover every requested shape, empty long history, short recent history,
    # padding assignments, finite backward, and learned-boundary gradients.
    for length, k in a.grid:
        modules = build_modules(a, 97, length, k, device)
        for module in modules:
            module.eval()
        seq = torch.zeros((4, length), dtype=torch.long)
        seq[0] = torch.randint(1, 98, (length,))
        seq[1, -73:] = torch.randint(1, 98, (73,))
        seq[2, -50:] = torch.randint(1, 98, (50,))
        seq[3, -3:] = torch.randint(1, 98, (3,))
        h, assignment, x = forward(modules, seq)
        assert h.shape == (4, a.hidden_dim) and assignment.shape == (4, length-50, k)
        assert torch.isfinite(h).all()
        assert torch.allclose(assignment.sum(-1), seq[:, :-50].ne(0).float(), atol=1e-5)
        assert assignment[2:].eq(0).all()
        # All slots stay exposed, but entirely empty rows cannot fabricate h_uk.
        zero_d = torch.zeros(4)
        representation, slots, _, _ = modules[1](x, seq[:, :-50].ne(0), zero_d)
        assert slots.all() and representation[2:].eq(0).all()
        # Masked arbitrary padding values cannot alter the subsequence encoder.
        poisoned = x.detach().clone()
        poisoned[~seq[:, :-50].ne(0)] = 1e4
        poison_h, _, poison_a, _ = modules[1](poisoned, seq[:, :-50].ne(0), zero_d)
        assert torch.allclose(representation, poison_h, atol=1e-5)
        assert torch.allclose(assignment, poison_a, atol=1e-6)
        compact = x[1:2, -23:]
        compact_h, _, compact_a, _ = modules[1](compact, torch.ones(1, 23, dtype=torch.bool), zero_d[:1])
        assert torch.allclose(representation[1:2], compact_h, atol=1e-5)
        assert torch.allclose(assignment[1:2, -23:], compact_a, atol=1e-6)
        l_struct = structure_loss(x, assignment)
        active_only = structure_loss(x[:2], assignment[:2])
        assert torch.allclose(l_struct, active_only, atol=1e-6), 'Empty rows dilute L_structure'
        empty_loss = structure_loss(x[2:], assignment[2:])
        assert empty_loss.item() == 0 and torch.isfinite(empty_loss)
        target = torch.tensor((1, 2, 3, 4))
        main = base.sampled_softmax_loss(h, target-1, modules[0].output_weight(), 11, a.temperature)
        (main+a.lambda_sub*l_struct).backward()
        params = [p for module in modules for p in module.parameters() if p.grad is not None]
        assert params and all(torch.isfinite(p.grad).all() for p in params)
        boundary_grad = modules[1].mlp_b[0].weight.grad
        assert boundary_grad is not None and boundary_grad.abs().sum() > 0
        assert modules[0].item_embedding.weight.grad.abs().sum() > 0
        assert modules[0].output_embedding.weight.grad.abs().sum() > 0
        # Check exact empty-long bypass, even if fusion has nonzero parameters.
        for cross in modules[2]:
            cross.gamma.data.fill_(1.0)
        modules[3].W_g.bias.data.fill_(10.0)
        with torch.no_grad():
            after, _, _ = forward(modules, seq[2:])
            model = modules[0]
            recent = seq[2:, -50:]
            padding = recent.eq(0)
            xx = model.item_embedding(recent)*math.sqrt(model.hidden_dim)
            xx = model.emb_dropout(xx+model.pos_embedding(torch.arange(50)[None, :]))
            xx = xx.masked_fill(padding[..., None], 0.0)
            for block in model.blocks:
                xx = block(xx, padding)
            xx = model.final_norm(xx).masked_fill(padding[..., None], 0.0)
            expected = F.normalize(xx[:, -1], dim=-1, eps=1e-6)
            assert torch.allclose(after, expected, atol=1e-6), 'Empty long modifies recent-only representation'
        results.append({'length': length, 'K': k, 'shape': list(assignment.shape),
                        'padding_invariance': True, 'empty_long_bypass': True,
                        'gradient_finite': True, 'boundary_gradient_nonzero': True,
                        'structure_ignores_empty_rows': True})
        print(f'CPU_CHECK L={length} K={k} PASS', flush=True)
        del modules
    # Exact chunked and dense search agree when scores have no ties.
    hidden, weights = torch.randn(7, 96), torch.randn(97, 96)
    expected = (hidden @ weights.T).topk(20, dim=1).indices+1
    for chunk in (1, 7, 19, 20, 37, 97, 128):
        actual = chunked_topk(hidden, weights, chunk)
        assert torch.equal(actual, expected), f'Chunked ranking differs for chunk={chunk}'
    return {'grid_checks': results, 'chunked_dense_top20': True,
            'chunk_sizes': [1, 7, 19, 20, 37, 97, 128],
            'cpu_offloaded_adamw_parity': optimizer_parity_check(),
            'single_gather_sampled_loss_parity': sampled_loss_parity_check(),
            'real_catalog_model_allocated': False}


def require_preflight(a, audit):
    preflight = json.loads(a.preflight_report.read_text(encoding='utf-8'))
    if preflight.get('passed') is not True:
        raise RuntimeError('CPU preflight has not passed')
    for key in ('manifest_sha256', 'files_sha256', 'source_hashes'):
        if preflight['input_audit'][key] != audit[key]:
            raise RuntimeError(f'Preflight stale: changed {key}')
    if preflight['model_signature'] != preflight_signature(a):
        raise RuntimeError('Preflight model/loss signature differs')
    covered = {(entry['length'], entry['K']) for entry in preflight['synthetic_checks']['grid_checks']}
    if not set(a.grid).issubset(covered):
        raise RuntimeError('Preflight did not cover every requested grid entry')
    return sha256(a.preflight_report)


def optimizer_for(a, modules):
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    kwargs = dict(lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=a.weight_decay)
    if a.optimizer_state_device == 'cpu':
        optimizer = CPUOffloadedAdamW(params, chunk_elements=a.optimizer_chunk_elements, **kwargs)
    else:
        optimizer = torch.optim.AdamW(params, foreach=False, **kwargs)
    return params, optimizer


def optimizer_state_bytes(optimizer):
    counts = {'cpu': 0, 'cuda': 0}
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                counts[value.device.type] = counts.get(value.device.type, 0)+value.numel()*value.element_size()
    return counts


def optimizer_config(a):
    return {'name': 'AdamW', 'state_device': a.optimizer_state_device,
            'chunk_elements': a.optimizer_chunk_elements if a.optimizer_state_device == 'cpu' else None,
            'lr': a.lr, 'betas': [0.9, 0.98], 'eps': 1e-8, 'weight_decay': a.weight_decay,
            'parameter_dtype': 'float32', 'moment_dtype': 'float32', 'amsgrad': False}


def training_step(a, modules, params, optimizer, seq, target):
    h, assignment, x = forward(modules, seq)
    main = sampled_loss(h, target, modules[0].output_embedding, a.num_negatives, a.temperature)
    struct = structure_loss(x, assignment)
    total = main+a.lambda_sub*struct
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite training loss')
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
    optimizer.step()
    return {'total': float(total.item()), 'main': float(main.item()),
            'structure': float(struct.item()), 'gradient_norm': float(grad_norm)}


def smoke_gpu(a, arrays, audit, device):
    """Real catalog allocation and optimizer state, with no formal result files."""
    if device.type != 'cuda':
        raise ValueError('--smoke-gpu requires CUDA')
    base.set_seed(a.seed)
    torch.cuda.reset_peak_memory_stats(device)
    # K20/L1000 maximizes this study's long-branch activation footprint.
    modules = build_modules(a, audit['num_items'], 1000, 20, device)
    params, optimizer = optimizer_for(a, modules)
    ds = LengthDataset(arrays['train'], 1000)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    steps = []
    for i, (seq, target) in enumerate(loader):
        if i >= 2:
            break
        for module in modules:
            module.train()
        steps.append(training_step(a, modules, params, optimizer, seq.to(device), target.to(device)))
        print(f'GPU_SMOKE step={i+1} losses={steps[-1]}', flush=True)
    if len(steps) != 2:
        raise ValueError('Smoke test requires at least two training batches')
    metrics = evaluate(a, modules, arrays['valid'], 1000, device, limit=min(a.eval_batch_size, 8))
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    result = {'passed': True, 'timestamp_utc': utc_now(), 'device': str(device),
              'num_items': audit['num_items'], 'steps': steps, 'mini_valid': metrics,
              'peak_memory_MB': torch.cuda.max_memory_allocated(device)/1024**2,
              'peak_reserved_MB': torch.cuda.max_memory_reserved(device)/1024**2,
              'device_free_MB_after': free/1024**2, 'device_total_MB': total/1024**2,
              'formal_training_started': False, 'smoke_optimizer_steps': 2,
              'optimizer': optimizer_config(a), 'optimizer_state_bytes': optimizer_state_bytes(optimizer),
              'input_manifest_sha256': audit['manifest_sha256'], 'source_hashes': sources()}
    write_json(a.output_dir/f'smoke_{str(device).replace(":", "")}.json', result)
    print('GPU_SMOKE PASS '+json.dumps(result), flush=True)
    return result


def train_one(a, arrays, audit, manifest, preflight_hash, length, k, device):
    run_dir = a.output_dir/f'L{length}_R50_K{k}_seed{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    base.set_seed(a.seed)
    modules = build_modules(a, audit['num_items'], length, k, device)
    params, optimizer = optimizer_for(a, modules)
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(LengthDataset(arrays['train'], length), batch_size=a.batch_size,
                        shuffle=True, generator=generator, num_workers=a.num_workers,
                        pin_memory=device.type == 'cuda')
    config = {key: portable_config(value) for key, value in vars(a).items()}
    config.update(length=length, recent_r=50, fixed_K=k, num_items=audit['num_items'],
                  parameter_count=sum(p.numel() for p in params),
                  d_u0=0, stage0_loaded=False, loss='L_main + 0.05 L_structure (default)',
                  time_features=False, time_loss=False, item_initialization='random',
                  selection_metric='valid NDCG@10', seen_filter=False,
                  input_manifest_sha256=audit['manifest_sha256'],
                  preflight_sha256=preflight_hash, source_hashes=sources(),
                  eval_protocol=audit['protocol'], catalog_mapping='item_mapping[dense_id-1]')
    config['optimizer'] = optimizer_config(a)
    write_json(run_dir/'config.json', config)
    write_json(run_dir/'input_manifest.json', manifest)
    write_json(run_dir/'input_audit.json', audit)
    best, best_epoch, stale = -math.inf, 0, 0
    started = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with open(run_dir/'train.log', 'x', encoding='utf-8') as stream:
        def log(message):
            line = f'[{utc_now()}] {message}'
            print(line, flush=True)
            stream.write(line+'\n'); stream.flush()
        log(f'START L={length} R50 K={k} seed={a.seed} V={audit["num_items"]}')
        for epoch in range(1, a.epochs+1):
            for module in modules:
                module.train()
            sums = {key: 0.0 for key in ('total', 'main', 'structure')}
            count = 0
            epoch_start = time.time()
            for step, (seq, target) in enumerate(loader, 1):
                step_started = time.time()
                seq, target = seq.to(device, non_blocking=True), target.to(device, non_blocking=True)
                loss = training_step(a, modules, params, optimizer, seq, target)
                count += len(seq)
                for key in sums:
                    sums[key] += loss[key]*len(seq)
                if step == 1 or step % a.log_every == 0:
                    log(f'epoch={epoch} step={step}/{len(loader)} step_seconds={time.time()-step_started:.3f} losses={json.dumps(loss)}')
            valid = evaluate(a, modules, arrays['valid'], length, device)
            score = valid['NDCG@10']
            entry = {'epoch': epoch, 'losses': {key: value/count for key, value in sums.items()},
                     'valid': valid, 'epoch_seconds': time.time()-epoch_start,
                     'train_samples': count, 'eval_samples': len(arrays['valid'][1])}
            with open(run_dir/'epochs.jsonl', 'a', encoding='utf-8') as epochs:
                epochs.write(json.dumps(entry, allow_nan=False)+'\n')
            log(f'EPOCH {json.dumps(entry)}')
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                # Save tensors directly: torch.save streams CUDA tensors through
                # CPU without retaining a second complete state dict on the CPU.
                state = {name: module.state_dict() for name, module in zip(NAMES, modules)}
                torch.save({'state': state, 'epoch': epoch, 'valid': valid, 'config': config},
                           run_dir/'best.pth.tmp')
                os.replace(run_dir/'best.pth.tmp', run_dir/'best.pth')
                del state
                log(f'BEST epoch={epoch} valid_NDCG@10={score:.9f}')
            else:
                stale += 1
                if a.early_stop_patience > 0 and stale >= a.early_stop_patience:
                    log('EARLY_STOP'); break
        # CPU loading avoids temporarily duplicating the full model on the GPU.
        try:
            checkpoint = torch.load(run_dir/'best.pth', map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(run_dir/'best.pth', map_location='cpu')
        for name, module in zip(NAMES, modules):
            module.load_state_dict(checkpoint['state'][name])
        best_valid = checkpoint['valid']
        del checkpoint
        test = evaluate(a, modules, arrays['test'], length, device)
        result = {'length': length, 'recent_r': 50, 'K': k, 'seed': a.seed,
                  'best_epoch': best_epoch, 'epochs_completed': epoch,
                  'best_valid_NDCG@10': best,
                  **{f'valid_{key}': value for key, value in best_valid.items()},
                  **{f'test_{key}': value for key, value in test.items()},
                  'num_items': audit['num_items'], 'actual_candidates': audit['num_items'],
                  'eval_samples': len(arrays['test'][1]), 'train_samples': len(arrays['train'][1]),
                  'seconds': time.time()-started, 'parameters': config['parameter_count'],
                  'peak_memory_MB': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None,
                  'directory': portable_path(run_dir), 'input_manifest_sha256': audit['manifest_sha256'],
                  'preflight_sha256': preflight_hash, 'protocol': audit['protocol'],
                  'catalog_mapping_sha256': audit['files_sha256']['item_mapping.npy'],
                  'source_hashes': audit['source_hashes'],
                  'optimizer': optimizer_config(a), 'optimizer_state_bytes': optimizer_state_bytes(optimizer),
                  'completed_utc': utc_now()}
        write_json(run_dir/'result.json', result)
        log('RESULT '+json.dumps(result))
    del modules, params, optimizer, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def save_summary(output_dir, results):
    write_json(output_dir/'summary.json', results)
    if results:
        with open(output_dir/'summary.csv.tmp', 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        os.replace(output_dir/'summary.csv.tmp', output_dir/'summary.csv')


def main():
    a = parse_args()
    torch.set_num_threads(a.cpu_threads)
    plan = {'dataset': 'Taobao MM', 'training_started': False, 'input_dir': portable_path(a.input_dir),
            'output_dir': portable_path(a.output_dir), 'device': a.device, 'recent_r': 50,
            'runs': [{'length': length, 'K': k, 'seed': a.seed} for length, k in a.grid],
            'loss': 'L_main + lambda_sub L_structure; no L_time; d0=0',
            'lambda_sub': a.lambda_sub, 'eval_seen_filter': False,
            'selection': 'validation NDCG@10', 'test': 'best checkpoint only',
            'note': 'No updates without --execute or the explicitly requested two-step --smoke-gpu.'}
    print(json.dumps(plan, indent=2), flush=True)
    if not (a.check or a.smoke_gpu or a.execute):
        return
    arrays, manifest, audit = load_input(a)
    if a.check:
        checks = synthetic_checks(a)
        report = {'passed': True, 'timestamp_utc': utc_now(), 'input_audit': audit,
                  'model_signature': preflight_signature(a), 'synthetic_checks': checks,
                  'real_training_started': False, 'real_optimizer_steps': 0}
        write_json(a.preflight_report, report)
        write_json(a.output_dir/'plan.json', plan)
        print(f'CPU_PREFLIGHT PASS report={a.preflight_report}', flush=True)
        return
    preflight_hash = require_preflight(a, audit)
    device = torch.device(a.device)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available():
        raise ValueError('An explicit available CUDA device is required for smoke/execute')
    torch.cuda.set_device(device)
    if a.smoke_gpu:
        smoke_gpu(a, arrays, audit, device)
        return
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plan['training_started'] = True
    write_json(a.output_dir/'plan.json', plan)
    write_json(a.output_dir/'audit.json', audit)
    results = []
    for length, k in a.grid:
        result = train_one(a, arrays, audit, manifest, preflight_hash, length, k, device)
        results.append(result)
        save_summary(a.output_dir, results)
    write_json(a.output_dir/'completed.json', {'completed': len(results), 'training_started': True,
                                               'completed_utc': utc_now(), 'runs': plan['runs']})
    print(f'COMPLETE {len(results)} runs: {a.output_dir}', flush=True)


if __name__ == '__main__':
    main()
