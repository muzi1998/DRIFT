# DRIFT-HSTU

Preference-drift-aware subsequence learning with an HSTU recommendation backbone.
The model encodes the recent window, attends to long-history subsequence
representations inside each HSTU block, and combines both contexts through an
output gate. This route scores real item IDs; the parent TIGER/GLASS route predicts
semantic IDs.

## Setup and path conventions

Run the examples from the parent repository root (`WWWcode`). A PyTorch environment
with `numpy`, `pandas`, `pyarrow`, and `tqdm` is required. Install the PyTorch build
appropriate for the available CPU/GPU; CUDA is used by the examples below.

```bash
python DRIFT-HSTU/run_hstu_drift_c_v5.py --help
python DRIFT-HSTU/run_hstu_drift_c_v5_taobaomm.py --help
```

Relative input, cache, and output arguments are anchored to the repository root by
`hstu_paths.py`, so changing into `DRIFT-HSTU` does not change their meaning. The
`data/` and `outputs/` prefixes also respect `DRIFT_DATA_ROOT` and
`DRIFT_OUTPUT_ROOT`, respectively. The examples contain no machine-specific paths.

## Entry points

| Files | Purpose |
|---|---|
| `run_hstu_drift_c_v5.py` | Main KuaiRec model with temporal and structure regularization |
| `run_hstu_drift_c_v5_taobaomm.py` | Main Taobao MM model without timestamp features |
| `run_hstu_drift_c_v5_ml20m.py` | ML-20M variant |
| `run_hstu_drift_c_v4*.py` | Corresponding variants without the v5 structure objective |
| `run_hstu_drift_c_v5_seqlen.py` | Controlled KuaiRec fixed-K/length study; extra SID Stage-0 assets required |
| `run_hstu_drift_c_v5_taobaomm_seqlen.py` | Controlled Taobao MM fixed-K/length study using restored item-ID arrays |
| `hstu_fixed_k.py`, `hstu_fixed_k_nots.py` | Fixed-K subsequence encoders for the controlled studies |
| `hstu_drift_common.py`, `run_hstu_kuairec.py` | Shared model/loss components and HSTU reference implementation |
| `stage0_common.py`, `stage0_model.py`, `stage0_losses.py`, `stage0_psi_features.py` | Item-level Stage-0 data, drift estimation, and objectives |
| `train_stage0_kuairec.py`, `train_stage0_ml20m.py`, `export_d_u_init_kuairec.py` | Stage-0 training and drift-cache export |
| `cpu_offloaded_adamw.py` | CPU optimizer-state option for the Taobao MM controlled study |
| `hstu_paths.py` | Repository-relative path handling |

### Meaning of K

The main v4/v5 entries use `--K_max`. It sets the number of allocated subsequence
slots and the upper bound of the effective count; it does **not** force every user
to have exactly K active subsequences. The effective count depends on the drift
input. The timestamp-free Taobao MM entry uses `d_u0=0`, which gives the midpoint
of the configured count interval.

Use the separate `*_seqlen.py` entries when an experiment requires **exactly**
K active slots. They preserve their own data and Stage-0 protocols and cannot be
substituted for the main entries by changing a filename alone.

## Data required by the main entries

The data and pretrained weights are not distributed with this code.

### KuaiRec

```text
data/kuairec_processed/
  train.parquet
  valid.parquet
  test.parquet
data/kuairec_caption_category.csv
```

Each interaction parquet contains `user_id`, `video_id`, and `timestamp`. The
loader merges the three interaction files, sorts by user/time, and maps item IDs
to a shared 1-based catalog (`0` is padding). The category CSV provides `video_id`
and `first_level_category_id` for Stage-0.

`build_glass_aligned_splits` uses the final two interactions as validation/test
targets. Both evaluations use the same preceding history. The last item of that
history is the training target; its prefix is the training input. Histories are
truncated and left-padded; `--max_len 1000` and `--recent_r 50` reserve up to 950
long-history positions and 50 recent positions.

### Taobao MM

```text
data/TAOBAO_MM/hstu_data/
  train.parquet
  valid.parquet
  test.parquet
```

Each parquet must contain `user`, `seq`, and `target`. `seq` is a list of real,
1-based integer item IDs and `target` is one integer item ID in the same catalog.
These are item-ID splits, not the three-level SID parquets used by TIGER. The
training rows provide the training history/target; validation rows provide the
history used for validation and test. Test rows provide the test target.

This dataset route has no timestamps or Stage-0 cache: it uses `d_u0=0` and omits
temporal features and the temporal structure term.

## Main experiments: L=1000, recent=50, K_max=10/20/30

### 1. Prepare the KuaiRec drift cache once

```bash
python DRIFT-HSTU/train_stage0_kuairec.py \
  --kuairec_data_dir data/kuairec_processed \
  --category_csv data/kuairec_caption_category.csv \
  --cache_dir outputs/hstu/drift_cache \
  --max_len 1000 --recent_r 50 --hidden_dim 96 \
  --epochs 8 --seed 2025 --device cuda:0
```

This produces `stage0_state.pt` and
`d_u_init_{train,valid,test}.parquet` in `outputs/hstu/drift_cache`. Reuse this cache
for the three K budgets with the same data, length, and recent-window settings.
The main KuaiRec entry requires the drift-cache parquets even when
`--load_stage0_emb` is omitted. That flag separately enables input-embedding
initialization from the checkpoint.

### 2. Run the three KuaiRec settings

The loop runs K_max=10, 20, and 30 sequentially, each with a separate output
folder:

```bash
for K in 10 20 30; do
  python DRIFT-HSTU/run_hstu_drift_c_v5.py \
    --kuairec_data_dir data/kuairec_processed \
    --drift_cache_dir outputs/hstu/drift_cache \
    --max_len 1000 --recent_r 50 --K_max "$K" \
    --batch_size 128 --eval_batch_size 128 \
    --epochs 100 --early_stop_patience 50 \
    --lambda_sub 0.05 --sub_lam_t 0.5 --sub_lam_c 0.5 \
    --selection_metric NDCG@10 --load_stage0_emb \
    --seed 2025 --device cuda:0 \
    --output_dir "outputs/hstu/kuairec_L1000_R50_K${K}"
done
```

For a single run, replace `"$K"` with `10`, `20`, or `30` and give that run its
own output directory.

### 3. Run the three Taobao MM settings

```bash
for K in 10 20 30; do
  python DRIFT-HSTU/run_hstu_drift_c_v5_taobaomm.py \
    --data_dir data/TAOBAO_MM/hstu_data \
    --max_len 1000 --recent_r 50 --K_max "$K" \
    --batch_size 128 --eval_batch_size 128 \
    --epochs 100 --early_stop_patience 50 \
    --lambda_sub 0.05 --selection_metric NDCG@20 \
    --seed 2025 --device cuda:0 \
    --output_dir "outputs/hstu/taobaomm_L1000_R50_K${K}"
done
```

The Taobao MM entry does not accept `--sub_lam_t`, `--sub_lam_c`, or
`--load_stage0_emb`; its objective is `L_main + lambda_sub * L_structure`.

### Objectives, outputs, and evaluation

For timestamped KuaiRec/ML-20M, v5 optimizes:

```text
L = L_main + lambda_sub * (sub_lam_t * L_time + sub_lam_c * L_structure)
```

`L_main` is sampled-softmax next-item cross-entropy. Evaluation ranks the complete
item catalog and reports Recall/NDCG at 1, 3, 5, 10, and 20. The main KuaiRec entry
excludes seen items while keeping the current target eligible; the Taobao MM
entry does not filter seen items. Model selection uses validation scores, and
the final test uses the best saved model.

Each main run writes `logs/*.log` and `ckpt/*.pth` under `--output_dir`. Artifact
stems include dataset, version, history length, recent window, K budget, and seed.
Use a fresh output folder for a new experiment; the legacy main trainers append
to their log file and can replace an existing checkpoint for the same setting.

Typical defaults differ between datasets: KuaiRec uses `lr=3e-4`, weight decay
`0`, batch size `128`, and selection `NDCG@10`; Taobao MM uses `lr=1e-4`, weight
decay `0.01`, batch size `512`, and selection `NDCG@20`. The examples explicitly
set batch size `128` for both datasets.

## Controlled studies with exactly K=10/20/30

These entries use a fixed recent window of 50 items. Their `--runs` argument
selects `(history length, fixed K)` pairs, and `1000:10`, `1000:20`, and `1000:30`
are supported. Without an execution flag they print a plan and do not train.

### KuaiRec fixed-K study

`run_hstu_drift_c_v5_seqlen.py` additionally requires:

- Aligned SID parquets named `train_seq_len1000_sim_augmented.parquet`,
  `valid_seq_len1000_with_history.parquet`, and
  `test_seq_len1000_with_history.parquet`, with `user`, `history`, `target`, and
  `timestamps_unix_s` columns.
- Source `item_embeddings.parquet` (`video_id`),
  `kuairec_rqvae_256_256_256.npy`, and `merged_all.parquet`
  (`user_id`, `video_id`, `timestamp`) to recover and verify real item identities.
- A compatible **SID** Stage-0 project exporting
  `drift_method.offline_drift.stage0_pretrainer.Stage0Model` and its frozen
  checkpoint. This optional dependency is not included here. The item-level
  Stage-0 cache from the main quickstart is a different model and cannot replace
  it.

After supplying these assets, run a CPU audit before training:

```bash
python DRIFT-HSTU/run_hstu_drift_c_v5_seqlen.py \
  --data_dir data/kuairec_processed/glass_data_256_256_256_aligned_days_ts \
  --source_root data/kuairec_processed \
  --stage0_project external/sid_stage0 \
  --stage0_state outputs/pretrained/sid_stage0_state.pt \
  --output_dir outputs/hstu/kuairec_fixed_k \
  --runs 1000:10 1000:20 1000:30 --seeds 2025 --device cpu --check
```

To train, use the same command with `--execute --device cuda:0` in place of
`--check --device cpu`. Source SIDs/timestamps/targets are audited before the
frozen Stage-0 model infers drift from each visible history.

### Taobao MM fixed-K study

This entry expects an already restored real-item dataset, not the main entry's
parquets. Its input directory contains:

```text
manifest.json                    # num_items and n_samples
item_mapping.npy                 # (num_items,), original embedding-row IDs
train_items.npy                  # int32 (n_samples, 1000), 0=left padding
valid_items.npy                  # same shape and catalog
test_items.npy                   # same shape and catalog
train_targets.npy                # int32 (n_samples,), IDs in 1..num_items
valid_targets.npy
test_targets.npy
```

All splits share a catalog; the arrays contain real item identities, not SID
triples. Create these assets during dataset preparation; an SID reverse lookup
is insufficient when multiple real items share a SID.

```bash
python DRIFT-HSTU/run_hstu_drift_c_v5_taobaomm_seqlen.py \
  --input_dir data/TAOBAO_MM/hstu_restored \
  --output_dir outputs/hstu/taobaomm_fixed_k \
  --runs 1000:10,1000:20,1000:30 --seed 2025 --check

python DRIFT-HSTU/run_hstu_drift_c_v5_taobaomm_seqlen.py \
  --input_dir data/TAOBAO_MM/hstu_restored \
  --output_dir outputs/hstu/taobaomm_fixed_k \
  --runs 1000:10,1000:20,1000:30 --seed 2025 --device cuda:0 --execute
```

The second command requires the matching preflight report produced by the first.
The controlled studies select validation `NDCG@10`, evaluate the best checkpoint
once on test, and do not filter seen items. They write plans, audits, checkpoints,
logs, and CSV/JSON summaries under their output directory. The Taobao MM study
can keep AdamW state on CPU through `--optimizer_state_device cpu`.
