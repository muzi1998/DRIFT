"""Controlled KuaiRec history-length / fixed-K experiment based on C-v5.

No training without --execute. --check audits all data and runs CPU forward-only
checks. The existing processed splits and targets are preserved; no target is
appended to its input. Raw video identities are recovered and verified because
SID triplets are not injective. Frozen SID Stage0 is used only to infer d_u from
the visible, truncated history, separately for every length and split.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from hstu_paths import resolve_args, portable_path, portable_config

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_hstu_drift_c_v5 as base
from hstu_fixed_k import FixedKSubseqRepresentationV5

HERE = Path(__file__).resolve().parent
GRID = ((100, 5), (100, 10), (500, 5), (500, 10), (500, 20),
        (1000, 5), (1000, 10), (1000, 20))
ALLOWED_GRID = GRID + ((1000, 30),)
FILES = {'train': 'train_seq_len1000_sim_augmented.parquet',
         'valid': 'valid_seq_len1000_with_history.parquet',
         'test': 'test_seq_len1000_with_history.parquet'}
TOPK = (1, 3, 5, 10, 20)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(portable_config(value), indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true', help='explicitly start training')
    modes.add_argument('--check', action='store_true', help='audit + CPU forward checks, no optimizer')
    p.add_argument('--data_dir', type=Path, default=Path('data/kuairec_processed/glass_data_256_256_256_aligned_days_ts'))
    p.add_argument('--source_root', type=Path, default=Path('data/kuairec_processed'))
    p.add_argument('--stage0_project', type=Path, default=Path('external/sid_stage0'))
    p.add_argument('--stage0_state', type=Path, default=Path('outputs/hstu/sid_stage0/stage0_state.pt'))
    p.add_argument('--output_dir', type=Path, default=Path('outputs/hstu/kuairec_fixed_k'))
    p.add_argument('--runs', nargs='+', help='optional subset, e.g. 100:5 500:20')
    p.add_argument('--seeds', type=int, nargs='+', default=[20260521])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--early_stop_patience', type=int, default=50)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--hidden_dim', type=int, default=96)
    p.add_argument('--linear_dim', type=int, default=16)
    p.add_argument('--attention_dim', type=int, default=16)
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--use_softmax_attn', action='store_true')
    p.add_argument('--num_negatives', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--lambda_sub', type=float, default=0.05)
    p.add_argument('--sub_lam_t', type=float, default=0.5)
    p.add_argument('--sub_lam_c', type=float, default=0.5)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--log_every', type=int, default=50)
    a = resolve_args(p.parse_args())
    if a.runs:
        try:
            a.grid = [tuple(map(int, r.split(':'))) for r in a.runs]
        except ValueError:
            p.error('--runs must use L:K')
        if any(r not in ALLOWED_GRID for r in a.grid) or len(set(a.grid)) != len(a.grid):
            p.error('--runs must be distinct supported L:K entries (also supports 1000:30)')
    else:
        a.grid = list(GRID)
    if len(set(a.seeds)) != len(a.seeds):
        p.error('seeds must be distinct')
    if min(a.batch_size, a.eval_batch_size, a.epochs, a.num_blocks, a.num_heads) < 1:
        p.error('batch sizes, epochs, blocks, heads must be positive')
    if a.num_negatives < 0 or a.num_workers < 0 or a.hidden_dim % 4:
        p.error('invalid negatives/workers, or hidden_dim not divisible by 4')
    if a.log_every < 1 or a.temperature <= 0:
        p.error('log_every and temperature must be positive')
    if min(a.lambda_sub, a.sub_lam_t, a.sub_lam_c) < 0:
        p.error('loss weights must be nonnegative')
    return a


def load_aligned_data(a):
    """Recover video IDs using the same source/order as rebuild_kuairec_timestamps.py.

    Each processed row is verified against *all* its source SIDs, Unix timestamps
    and target before being admitted. No reverse lookup of non-unique SIDs.
    """
    video_ids = pd.read_parquet(a.source_root / 'item_embeddings.parquet', columns=['video_id'])['video_id'].to_numpy(np.int64)
    codes = np.load(a.source_root / 'kuairec_rqvae_256_256_256.npy')[:, :3].astype(np.int64)
    if len(video_ids) != len(codes) or len(np.unique(video_ids)) != len(video_ids):
        raise ValueError('Video/SID catalog shape or video uniqueness mismatch')
    raw = pd.read_parquet(a.source_root / 'merged_all.parquet', columns=['user_id', 'video_id', 'timestamp'])
    item_rows = pd.Index(video_ids).get_indexer(raw['video_id'].to_numpy())
    raw = raw.loc[item_rows >= 0].copy()
    raw['item_row'] = item_rows[item_rows >= 0]
    raw = raw.sort_values(['user_id', 'timestamp'])  # same tie ordering as preprocessing
    raw_users = raw['user_id'].to_numpy(np.int64)
    raw_items = raw['item_row'].to_numpy(np.int64)
    raw_ts = raw['timestamp'].to_numpy(np.float64)
    if not np.isfinite(raw_ts).all() or np.any(raw_ts <= 0):
        raise ValueError('Source timestamps must be finite positive Unix seconds')
    users, starts, counts = np.unique(raw_users, return_index=True, return_counts=True)
    slices = {int(u): (int(st), int(st+n)) for u, st, n in zip(users, starts, counts)}
    del raw, item_rows, raw_users
    out, audit = {}, {'splits': {}, 'num_items': len(video_ids),
                      'unique_sid_triplets': len(np.unique(codes, axis=0)),
                      'item_mapping_sha256': hashlib.sha256(video_ids.tobytes() + codes.tobytes()).hexdigest()}
    fingerprints = {}
    for split, filename in FILES.items():
        path = a.data_dir / filename
        table = pq.read_table(path, columns=['user', 'history', 'target', 'timestamps_unix_s'])
        n = table.num_rows
        items = np.zeros((n, 1000), dtype=np.int64)
        ts = np.zeros((n, 1000), dtype=np.float64)
        targets, user_list, lengths = [], [], []
        for i, row in enumerate(table.to_pylist()):
            uid = int(row['user'])
            if uid not in slices:
                raise ValueError(f'{split}: source user missing {uid}')
            lo, hi = slices[uid]
            source_items, source_ts = raw_items[lo:hi], raw_ts[lo:hi]
            if len(source_items) < 4:
                raise ValueError(f'{split}: insufficient source history for {uid}')
            visible, visible_ts = source_items[:-2][-1000:], source_ts[:-2][-1000:]
            if split == 'train':
                hist, hist_ts, target = visible[:-1], visible_ts[:-1], visible[-1]
            else:
                hist, hist_ts = visible, visible_ts
                target = source_items[-2 if split == 'valid' else -1]
            actual = np.asarray(row['history'], dtype=np.int64).reshape(-1, 3)
            if not np.array_equal(actual, codes[hist]) or not np.array_equal(row['target'], codes[target]):
                raise ValueError(f'{split}: SID history/target mismatch for user {uid}')
            padded = np.asarray(row['timestamps_unix_s'], dtype=np.float64)
            expected_ts = np.zeros(1000, dtype=np.float64)
            expected_ts[-len(hist):] = hist_ts
            if padded.shape != (1000,) or not np.array_equal(padded, expected_ts):
                raise ValueError(f'{split}: history/timestamp mismatch for user {uid}')
            if len(hist) <= 50 or np.any(np.diff(hist_ts) < 0):
                raise ValueError(f'{split}: requires >50 historical items and monotonic times; user {uid}')
            items[i, -len(hist):] = hist + 1  # actual embedding row identity, PAD=0
            ts[i] = padded
            targets.append(int(target) + 1)
            user_list.append(uid)
            lengths.append(len(hist))
        if len(set(user_list)) != n:
            raise ValueError(f'{split}: duplicate users')
        out[split] = {'items': items, 'unix_ts': ts, 'targets': np.array(targets, dtype=np.int64),
                      'users': np.array(user_list, dtype=np.int64)}
        fp = hashlib.sha256(out[split]['users'].tobytes() + out[split]['targets'].tobytes()).hexdigest()
        audit['splits'][split] = {'rows': n, 'min_input_length': min(lengths),
                                  'max_input_length': max(lengths), 'users_targets_sha256': fp,
                                  'source_sha256': sha256(path)}
        fingerprints[split] = set(user_list)
        print(f'AUDIT {split}: {n} rows; input lengths {min(lengths)}..{max(lengths)}; all SID/time/target checks passed', flush=True)
    if not (fingerprints['train'] == fingerprints['valid'] == fingerprints['test']):
        raise ValueError('Split user cohorts differ; refusing to silently select users')
    if len(video_ids) < max(TOPK):
        raise ValueError('Catalog is too small for top-k evaluation')
    audit['protocol'] = 'original processed splits/targets; train history excludes its target; full 1000-slot history masks fixed across L'
    return out, codes, video_ids, audit


def load_stage0(a):
    project = str(a.stage0_project.resolve())
    sys.path.insert(0, project)
    module_path = a.stage0_project / 'drift_method/offline_drift/stage0_pretrainer.py'
    if not module_path.is_file():
        raise FileNotFoundError(
            'The fixed-K KuaiRec runner needs the optional frozen SID Stage0 '
            'package under --stage0_project (drift_method/offline_drift/'
            'stage0_pretrainer.py). The item-level stage0_model.py is a '
            'different model. Use run_hstu_drift_c_v5.py with its local Stage0 '
            'cache for the self-contained adaptive route.')
    module = importlib.import_module('drift_method.offline_drift.stage0_pretrainer')
    state = torch.load(a.stage0_state, map_location='cpu')
    cfg = state['config']
    if cfg.get('target_depth') != 3 or cfg.get('sid1_size') != 256 or cfg.get('pad_token_id') != 0 or cfg.get('recent_window_r') != 50:
        raise ValueError('Expected frozen KuaiRec SID Stage0 with depth=3, codebook=256, PAD=0, recent_r=50')
    model = module.Stage0Model(**cfg)
    model.load_export_state(state)
    model.eval().requires_grad_(False)
    return model, {'path': portable_path(a.stage0_state), 'sha256': sha256(a.stage0_state),
                   'config': cfg, 'module_sha256': sha256(module.__file__),
                   'features_sha256': sha256(Path(module.__file__).with_name('psi_features.py'))}


def make_visible(data, length):
    items = data['items'][:, -length:].copy()
    unix_ts = data['unix_ts'][:, -length:].copy()
    valid = items != 0
    # Float64 difference first; float32 absolute Unix times lose sub-minute gaps.
    dt = np.zeros_like(unix_ts)
    pair = valid[:, 1:] & valid[:, :-1]
    dt[:, 1:] = np.where(pair, np.diff(unix_ts, axis=1), 0.0)
    log_dt = np.log1p(np.maximum(dt, 0)).astype(np.float32)
    first = np.where(valid, unix_ts, np.inf).min(axis=1, keepdims=True)
    days = np.where(valid, (unix_ts-first)/86400.0, 0.0).astype(np.float32)
    return items, log_dt, days


def infer_drift(model, items, days, codes, device, batch_size):
    model = model.to(device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            block = items[start:start+batch_size]
            sid = codes[np.maximum(block-1, 0)]
            tokens = sid + np.array([1, 257, 513], dtype=np.int64)
            tokens[block == 0] = 0
            token_tensor = torch.from_numpy(tokens.reshape(len(block), -1)).to(device)
            time_tensor = torch.from_numpy(days[start:start+batch_size]).to(device)
            drift, _, _, _ = model(token_tensor, history_ts=time_tensor, recent_window_r=50)
            if not torch.isfinite(drift).all():
                raise ValueError('Stage0 produced nonfinite drift')
            output.append(drift.cpu())
    return torch.cat(output)


def build_modules(a, n_items, length, k, device):
    model = base.HSTUV4(n_items, 50, a.hidden_dim, a.linear_dim, a.attention_dim,
                        a.num_blocks, a.num_heads, a.dropout, a.attn_dropout,
                        use_softmax=a.use_softmax_attn).to(device)
    subseq = FixedKSubseqRepresentationV5(a.hidden_dim, K_max=k, max_long_len=length-50).to(device)
    huk = nn.ModuleList([base.HukCrossAttnV2(a.hidden_dim, num_heads=4, dropout=0.1)
                         for _ in range(a.num_blocks)]).to(device)
    gate = base.OutputGate(a.hidden_dim).to(device)
    return model, subseq, huk, gate


def forward(modules, seq, du, log_dt):
    return base.forward_hstu_drift_c_v4(*modules, seq, du, log_dt, 50)


def evaluate(modules, tensors, full_data, device, batch_size):
    for module in modules:
        module.eval()
    totals = {f'{m}@{k}': 0.0 for m in ('Recall', 'NDCG') for k in TOPK}
    # Candidate masks always use original 1000-slot history, independent of L.
    with torch.inference_mode():
        for start in range(0, len(tensors[0]), batch_size):
            seq, target, du, log_dt = (x[start:start+batch_size].to(device) for x in tensors)
            h, _, _, _ = forward(modules, seq, du, log_dt)
            scores = h @ modules[0].output_weight().t()
            for j in range(len(seq)):
                history = full_data['items'][start+j]
                masked = history[(history != 0) & (history != int(target[j].item()))]
                if len(masked):
                    scores[j, torch.as_tensor(masked-1, dtype=torch.long, device=device)] = -torch.inf
            predictions = scores.topk(max(TOPK), dim=1).indices + 1
            hits = predictions.eq(target[:, None])
            for k in TOPK:
                hit = hits[:, :k]
                totals[f'Recall@{k}'] += hit.any(dim=1).sum().item()
                discount = 1.0 / torch.log2(torch.arange(k, device=device, dtype=torch.float32)+2)
                totals[f'NDCG@{k}'] += (hit.float()*discount).sum().item()
    return {key: value/len(tensors[0]) for key, value in totals.items()}


def forward_checks(a, n_items, codes, datasets, stage0):
    """CPU, inference only: no loss.backward(), optimizer, or training loop."""
    torch.set_num_threads(2)
    device = torch.device('cpu')
    checks = []
    for length, k in a.grid:
        base.set_seed(a.seeds[0])
        modules = build_modules(a, n_items, length, k, device)
        for module in modules:
            module.eval()
        for split, block in datasets.items():
            items, log_dt, days = make_visible(block, length)
            with torch.inference_mode():
                du = infer_drift(stage0, items[:2], days[:2], codes, device, 2)
                seq, ld = torch.from_numpy(items[:2]), torch.from_numpy(log_dt[:2])
                h, assign, x, long_dt = forward(modules, seq, du, ld)
                _, mask, _, _ = modules[1](modules[0].item_embedding(seq[:, :-50])*math.sqrt(a.hidden_dim),
                                           seq[:, :-50].ne(0), du, ld[:, :-50])
                sub, lt, lc = base.compute_l_sub_v5(assign, x, long_dt, a.sub_lam_t, a.sub_lam_c)
                assert h.shape == (2, a.hidden_dim) and assign.shape == (2, length-50, k)
                assert bool(mask.all()), 'Fixed-K must expose every K slot'
                assert bool(torch.isfinite(h).all()) and bool(torch.isfinite(sub))
                assert torch.allclose(assign.sum(-1), seq[:, :-50].ne(0).float(), atol=1e-5)
                checks.append({'length': length, 'K': k, 'split': split, 'shape': list(assign.shape),
                               'finite': True, 'fixed_K': True, 'sample_L_sub': float(sub)})
        print(f'FORWARD_ONLY L={length} K={k}: PASS', flush=True)
        del modules
    return checks


def train_one(a, n_items, length, k, seed, visible, full_data, suite_dir):
    run_dir = suite_dir / f'L{length}_R50_K{k}_seed{seed}'
    run_dir.mkdir(exist_ok=False)
    device = torch.device(a.device)
    base.set_seed(seed)
    modules = build_modules(a, n_items, length, k, device)
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=a.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*visible['train']), batch_size=a.batch_size, shuffle=True,
                        generator=generator, num_workers=a.num_workers, pin_memory=device.type == 'cuda')
    config = {**{key: portable_config(value) for key, value in vars(a).items()},
              'length': length, 'recent_r': 50, 'fixed_K': k, 'seed': seed,
              'num_items': n_items, 'input_embedding_init': 'random, as v5 default',
              'stage0_role': 'frozen SID model for d_u only; recomputed per visible L/split',
              'parameter_count': sum(p.numel() for p in params)}
    write_json(run_dir / 'config.json', config)
    best, best_epoch, stale = -math.inf, None, 0
    start_time = time.time()
    with open(run_dir / 'train.log', 'x', encoding='utf-8') as log_file:
        def log(message):
            message = f'[{datetime.now().isoformat(timespec="seconds")}] {message}'
            print(message, flush=True)
            log_file.write(message+'\n'); log_file.flush()
        log(f'START L={length} R=50 fixed_K={k} seed={seed}')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, a.epochs+1):
            for module in modules:
                module.train()
            sums = np.zeros(4, dtype=np.float64)
            samples = 0
            for step, batch in enumerate(loader, 1):
                seq, target, du, log_dt = (v.to(device) for v in batch)
                h, assign, x, long_dt = forward(modules, seq, du, log_dt)
                main = base.sampled_softmax_loss(h, target-1, modules[0].output_weight(), a.num_negatives, a.temperature)
                sub, lt, lc = base.compute_l_sub_v5(assign, x, long_dt, a.sub_lam_t, a.sub_lam_c)
                loss = main + a.lambda_sub*sub
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss L={length} K={k} epoch={epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
                optimizer.step()
                samples += len(seq)
                sums += np.array([loss.item(), main.item(), lt.item(), lc.item()])*len(seq)
                if step % a.log_every == 0:
                    log(f'epoch={epoch} step={step} total={loss.item():.5f} main={main.item():.5f} time={lt.item():.5f} structure={lc.item():.5f}')
            val = evaluate(modules, visible['valid'], full_data['valid'], device, a.eval_batch_size)
            score = val['NDCG@10']
            log(f'epoch={epoch} mean_losses={list(sums/samples)} valid={json.dumps(val)}')
            with open(run_dir / 'epochs.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'epoch': epoch, 'losses': (sums/samples).tolist(), 'valid': val})+'\n')
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                state = {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                         for name, module in zip(('model', 'subseq', 'huk', 'gate'), modules)}
                torch.save({'state': state, 'epoch': epoch, 'valid': val, 'config': config}, run_dir / 'best.pth')
            else:
                stale += 1
                if a.early_stop_patience > 0 and stale >= a.early_stop_patience:
                    log('EARLY_STOP'); break
        checkpoint = torch.load(run_dir / 'best.pth', map_location='cpu')
        for name, module in zip(('model', 'subseq', 'huk', 'gate'), modules):
            module.load_state_dict(checkpoint['state'][name])
        test = evaluate(modules, visible['test'], full_data['test'], device, a.eval_batch_size)
        result = {'length': length, 'recent_r': 50, 'K': k, 'seed': seed, 'best_epoch': best_epoch,
                  'best_valid_NDCG@10': best, **{f'test_{key}': value for key, value in test.items()},
                  'seconds': time.time()-start_time, 'parameters': config['parameter_count'],
                  'peak_memory_MB': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None,
                  'directory': portable_path(run_dir)}
        write_json(run_dir / 'result.json', result)
        log('RESULT '+json.dumps(result))
    del checkpoint, modules, optimizer, params, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    a = parse_args()
    plan = {'training_started': False, 'recent_r': 50, 'K_mode': 'fixed',
            'runs': [{'length': l, 'long_slots': l-50, 'K': k, 'seed': s} for s in a.seeds for l, k in a.grid],
            'selection_metric': 'validation NDCG@10', 'test_policy': 'best checkpoint only',
            'data_dir': portable_path(a.data_dir), 'stage0_state': portable_path(a.stage0_state),
            'note': 'No training without --execute. --check performs CPU inference only.'}
    print(json.dumps(plan, indent=2), flush=True)
    if not a.check and not a.execute:
        return
    data, codes, video_ids, audit = load_aligned_data(a)
    stage0, stage0_meta = load_stage0(a)
    audit['stage0'] = stage0_meta
    audit['sources'] = {name: sha256(HERE/name) for name in ('run_hstu_drift_c_v5.py', 'hstu_fixed_k.py', Path(__file__).name,
                                                          'run_hstu_kuairec.py', 'stage0_common.py')}
    audit['raw_sources'] = {name: sha256(a.source_root/name) for name in
                            ('item_embeddings.parquet', 'kuairec_rqvae_256_256_256.npy', 'merged_all.parquet')}
    if a.check:
        audit['forward_checks'] = forward_checks(a, len(video_ids), codes, data, stage0)
        audit['training_started'] = False
        write_json(a.output_dir / 'preflight.json', audit)
        write_json(a.output_dir / 'plan.json', plan)
        print(f'CHECK COMPLETE; no optimizer/training; report={a.output_dir / "preflight.json"}', flush=True)
        return
    if a.device != 'cpu' and (not torch.cuda.is_available() or torch.device(a.device).index is None):
        raise RuntimeError('Explicit available CUDA device (e.g. cuda:0), or cpu, required')
    suite_dir = a.output_dir / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    suite_dir.mkdir(parents=True, exist_ok=False)
    plan['training_started'] = True
    write_json(suite_dir / 'audit.json', audit)
    write_json(suite_dir / 'plan.json', plan)
    pd.DataFrame({'embedding_id': np.arange(1, len(video_ids)+1), 'video_id': video_ids}).to_csv(suite_dir / 'item_mapping.csv', index=False)
    results = []
    for length in sorted({l for l, _ in a.grid}):
        visible = {}
        for split, block in data.items():
            items, log_dt, days = make_visible(block, length)
            du = infer_drift(stage0, items, days, codes, torch.device(a.device), a.eval_batch_size)
            visible[split] = (torch.from_numpy(items), torch.from_numpy(block['targets']), du, torch.from_numpy(log_dt))
            np.savez_compressed(suite_dir / f'drift_L{length}_{split}.npz', users=block['users'], d_u=du.numpy())
        stage0.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for seed in a.seeds:
            for l, k in a.grid:
                if l != length:
                    continue
                result = train_one(a, len(video_ids), length, k, seed, visible, data, suite_dir)
                results.append(result)
                pd.DataFrame(results).to_csv(suite_dir / 'summary.csv', index=False)
                write_json(suite_dir / 'summary.json', results)
        del visible
        gc.collect()
    frame = pd.DataFrame(results)
    metrics = [c for c in frame.columns if c.startswith('test_')]
    frame.groupby(['length', 'K'])[metrics].agg(['mean', 'std']).to_csv(suite_dir / 'summary_mean_std.csv')
    write_json(suite_dir / 'completed.json', {'completed': len(results), 'training_started': True})
    print(f'COMPLETE {len(results)} runs: {suite_dir}', flush=True)


if __name__ == '__main__':
    main()
