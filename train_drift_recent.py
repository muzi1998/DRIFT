#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the revised DRIFT-TIGER with a recent-window encoder prefix.

The reviewed decoder fuses long-history subsequences at every SID position.
Partition statistics ignore padding, and evaluation assignments are deterministic.
Data and output paths use the repository-relative ``paths`` helpers.

Example with aligned day timestamps::

    python train_drift_recent.py \
        --dataset_dir kuairec_processed/glass_data_256_256_256_aligned_days_ts \
        --max_len 1000 --recent_window_r 50 --K 10 \
        --batch_size 128 --infer_size 64 --timestamp_unit days \
        --out_dir outputs/drift_recent_revised

Use ``--timestamp_unit seconds`` for Unix-second ``timestamps`` columns, or
``--use_timestamps 0`` for datasets without timestamps. See README.md for the
entry interfaces, dataset layouts, and differences from the original variant.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import paths

paths.ensure_import_paths()

import train_core
from drift_method.revised import RevisedDrift
from model.revised_data import RevisedDataset


VARIANT = 'drift_recent_revised'
TOPK = (1, 3, 5, 10, 20)


def build_parser():
    ap = argparse.ArgumentParser(
        description='Revised DRIFT-TIGER; encoder prefix = [recent window]')
    train_core.add_common_arguments(ap)
    # Defaults of the reviewed quality experiment; each remains configurable.
    ap.set_defaults(max_len=1000, recent_window_r=50, K=10,
                    batch_size=128, infer_size=64)
    for action in ap._actions:
        if action.dest == 'rec_item_repr':
            action.choices = ['first']
            action.help = 'reviewed target representation: first-level SID'
    ap.add_argument('--dataset_dir', default=None,
                    help='processed directory relative to the data root; '
                         'overrides data_root/codebook-directory construction')
    ap.add_argument('--train_file', default=None,
                    help='optional training parquet name inside the dataset directory')
    ap.add_argument('--valid_file', default=None,
                    help='optional validation parquet name inside the dataset directory')
    ap.add_argument('--test_file', default=None,
                    help='optional test parquet name inside the dataset directory')
    ap.add_argument('--timestamp_unit', choices=['days', 'seconds'], default='days',
                    help='unit of the timestamps column; model inputs are relative days')
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:<index>')
    ap.add_argument('--num_threads', type=int, default=4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--out_dir', default=paths.output_path(VARIANT),
                    help='directory for the best checkpoint and metric JSON files')
    return ap


def validate_args(args):
    try:
        codebook = tuple(int(x.strip()) for x in args.codebook.split(','))
    except ValueError as exc:
        raise ValueError('--codebook expects three positive integer sizes') from exc
    if len(codebook) != 3 or min(codebook) < 1:
        raise ValueError('--codebook expects three positive integer sizes')
    if not 1 <= args.recent_window_r < args.max_len:
        raise ValueError('Require 1 <= recent_window_r < max_len')
    k_max = float(args.K if args.K_max <= 0 else args.K_max)
    if not 1 <= args.K_min <= k_max <= args.K:
        raise ValueError('Require 1 <= K_min <= K_max <= K')
    for name in ('batch_size', 'infer_size', 'num_epochs', 'beam_size', 'num_threads'):
        if getattr(args, name) < 1:
            raise ValueError('--' + name + ' must be positive')
    if args.pad_token_id != 0:
        raise ValueError('The revised SID data format uses pad_token_id=0')
    if args.use_timestamps not in (0, 1):
        raise ValueError('--use_timestamps must be 0 or 1')
    if args.rec_item_repr != 'first':
        raise ValueError('The reviewed recommendation loss uses rec_item_repr=first')
    if args.rec_tau <= 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError('rec_tau, lr and grad_clip must be positive')
    for name in ('lambda_rec', 'lambda_subrec', 'lambda_sub', 'sub_lam_t',
                 'sub_lam_c', 'weight_decay'):
        if getattr(args, name) < 0 or not math.isfinite(getattr(args, name)):
            raise ValueError('--' + name + ' must be finite and nonnegative')
    for name in ('rec_tau', 'lr', 'grad_clip', 'K_min', 'K_max'):
        if not math.isfinite(getattr(args, name)):
            raise ValueError('--' + name + ' must be finite')
    return codebook


def configure(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_dataset_path(args, codebook):
    if args.dataset_dir is None:
        return train_core.resolve_dataset_path(args, codebook)
    result = paths.data_path(args.dataset_dir)
    if not os.path.isdir(result):
        raise FileNotFoundError('Processed dataset directory does not exist; '
                                'check --dataset_dir or DRIFT_DATA_ROOT')
    return result


def split_files(args, dataset_path):
    names = {'train': 'train_sim_augmented.parquet',
             'valid': 'valid_with_history.parquet',
             'test': 'test_with_history.parquet'}
    result = {}
    for split, default in names.items():
        explicit = getattr(args, split + '_file')
        if explicit is not None:
            # Split flags select filenames; the dataset location has its own flag.
            if Path(explicit).name != explicit or '\\' in explicit:
                raise ValueError('--' + split + '_file must be a filename')
            result[split] = os.path.join(dataset_path, explicit)
        else:
            result[split] = train_core.resolve_split_file(
                dataset_path, default, args.max_len)
    return result


def build_model(args, codebook, dataset_path, device):
    config = train_core.build_model_config(args, codebook, dataset_path, device)
    stride = max(codebook)
    highest_sid = max(i * stride + size for i, size in enumerate(codebook))
    config.update(use_gate=False, use_sidtier=False,
                  bos_token_id=highest_sid + 1,
                  vocab_size=max(1000, highest_sid + 2))
    # Avoid carrying machine-dependent paths into checkpoints and summaries.
    config.pop('dataset_path', None)
    if args.beam_size > config['vocab_size']:
        raise ValueError('beam_size must not exceed vocabulary size')
    cfg = train_core.build_drift_cfg(args, codebook, device)
    cfg.time_unit_scale = 1.0
    model = RevisedDrift(
        config, cfg, lambda_rec=args.lambda_rec, lambda_subrec=args.lambda_subrec,
        lambda_sub=args.lambda_sub, sub_lam_t=args.sub_lam_t,
        sub_lam_c=args.sub_lam_c, rec_tau=args.rec_tau).to(device)
    return model, config, cfg


def make_loader(dataset, batch_size, shuffle=False, seed=2025):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False, generator=generator)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    sums = {key: 0.0 for key in ('total', 'ce', 'rec', 'subrec', 'structure')}
    samples = 0
    for batch in tqdm(loader, desc='  training', leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        terms = model.loss_components(batch)
        if any(not torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError('Non-finite training loss')
        terms['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradient norm')
        optimizer.step()
        count = batch['target'].size(0)
        for key in sums:
            sums[key] += float(terms[key].detach()) * count
        samples += count
    if not samples:
        raise ValueError('Training split is empty')
    return {key: value / samples for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, loader, beam_size, device):
    model.eval()
    sums = {f'{kind}@{k}': 0.0 for kind in ('Recall', 'NDCG') for k in TOPK}
    prefix = torch.zeros(3, device=device, dtype=torch.float64)
    samples = 0
    for batch in tqdm(loader, desc='  evaluating', leave=False):
        batch = move_batch(batch, device)
        pred = model.generate(batch, beam_size=beam_size, max_length=4)
        target = batch['target']
        matches = (pred == target[:, None, :]).all(-1)
        ranks = torch.arange(1, pred.size(1) + 1, device=device).expand_as(matches)
        # The fourth generated token can yield duplicate three-token prefixes.
        # Count only the first matching beam for both Recall and NDCG.
        first = ranks.masked_fill(~matches, pred.size(1) + 1).min(1).values
        for k in TOPK:
            hit = first <= min(k, pred.size(1))
            sums[f'Recall@{k}'] += float(hit.sum())
            gain = torch.where(hit, 1.0 / torch.log2(first.float() + 1.0),
                               torch.zeros_like(first, dtype=torch.float32))
            sums[f'NDCG@{k}'] += float(gain.sum())
        prefix += (pred[:, 0, :] == target).long().cumprod(-1).sum(0)
        samples += target.size(0)
    if not samples:
        raise ValueError('Evaluation split is empty')
    probabilities = prefix.cpu().tolist()
    return {**{key: value / samples for key, value in sums.items()},
            'P_1': probabilities[0] / samples,
            'P_1_2': probabilities[1] / samples,
            'P_1_2_3': probabilities[2] / samples, 'samples': samples}


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def run_training(args):
    codebook = validate_args(args)
    configure(args.seed, args.num_threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    dataset_path = resolve_dataset_path(args, codebook)
    files = split_files(args, dataset_path)
    out = Path(args.out_dir)
    best_json = out / (VARIANT + '_best.json')
    summary_json = out / (VARIANT + '_summary.json')
    checkpoint_path = out / (VARIANT + '_best.pt')
    for item in (best_json, summary_json, checkpoint_path):
        if item.exists() or item.with_name(item.name + '.tmp').exists():
            raise FileExistsError('Run artifacts already exist; choose a fresh --out_dir')
    datasets = {
        split: RevisedDataset(filename, codebook, args.max_len,
                              use_timestamps=bool(args.use_timestamps),
                              timestamp_unit=args.timestamp_unit)
        for split, filename in files.items()
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError('All three data splits must contain samples')
    out.mkdir(parents=True, exist_ok=True)
    model, model_config, cfg = build_model(args, codebook, dataset_path, device)
    # The composite module exposes the tied embedding only once.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    valid_dl = make_loader(datasets['valid'], args.infer_size)
    test_dl = make_loader(datasets['test'], args.infer_size)
    logging.info('[%s] device=%s L=%s r=%s K=%s data=%s', VARIANT, device,
                 args.max_len, args.recent_window_r, args.K, Path(dataset_path).name)
    logging.info('[data] %s', ' '.join(
        f'{split}={Path(filename).name}({len(datasets[split])})'
        for split, filename in files.items()))
    history = []
    best_score, best_epoch, stale = -float('inf'), 0, 0
    started = time.monotonic()
    for epoch in range(1, args.num_epochs + 1):
        train_dl = make_loader(datasets['train'], args.batch_size, True,
                               args.seed + epoch)
        losses = train_one_epoch(model, train_dl, optimizer, device, args.grad_clip)
        valid = evaluate(model, valid_dl, args.beam_size, device)
        history.append({'epoch': epoch, 'train': losses, 'valid': valid})
        logging.info('[Epoch %s] loss=%.6f CE=%.6f rec=%.6f subrec=%.6f '
                     'structure=%.6f valid NDCG@20=%.6f', epoch, losses['total'],
                     losses['ce'], losses['rec'], losses['subrec'],
                     losses['structure'], valid['NDCG@20'])
        if valid['NDCG@20'] > best_score:
            best_score, best_epoch, stale = valid['NDCG@20'], epoch, 0
            temporary = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'valid': valid, 'model_config': model_config,
                        'drift_config': dict(vars(cfg)), 'variant': VARIANT}, temporary)
            temporary.replace(checkpoint_path)
            save_json(best_json, {'best_epoch': epoch, 'valid': valid})
        else:
            stale += 1
        if args.early_stop > 0 and stale >= args.early_stop:
            logging.info('[early stop] epoch=%s', epoch)
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    test = evaluate(model, test_dl, args.beam_size, device)
    best = {'best_epoch': best_epoch, 'best_valid_ndcg@20': best_score,
            'valid': checkpoint['valid'], 'test': test,
            'checkpoint': checkpoint_path.name}
    save_json(best_json, best)
    summary = {'variant': VARIANT, 'best': best, 'epochs_run': len(history),
               'codebook': codebook, 'K': args.K, 'max_len': args.max_len,
               'recent_window_r': args.recent_window_r,
               'loss_weights': {name: getattr(args, name) for name in
                                ('lambda_rec', 'lambda_subrec', 'lambda_sub',
                                 'sub_lam_t', 'sub_lam_c', 'rec_tau')},
               'timestamp_input_unit': args.timestamp_unit,
               'timestamp_model_unit': 'days',
               'files': {key: Path(value).name for key, value in files.items()},
               'model_config': model_config, 'drift_config': dict(vars(cfg)),
               'history': history, 'elapsed_sec': round(time.monotonic() - started, 1)}
    save_json(summary_json, summary)
    logging.info('[best epoch=%s] final test %s', best_epoch, json.dumps(test))
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    run_training(build_parser().parse_args())


if __name__ == '__main__':
    main()
