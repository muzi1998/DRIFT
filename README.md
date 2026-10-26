# Preference-Drift-Aware Subsequence Learning for Long-Sequence Recommendation

This package contains generative DRIFT-TIGER and DRIFT-GLASS implementations, an HSTU implementation, and preprocessing and semantic-ID utilities. Datasets and trained checkpoints are supplied separately.

## 1. Environment

The training code requires Python, PyTorch, NumPy, pandas, PyArrow, and tqdm. The preprocessing and semantic-ID utilities additionally use Transformers, sentence-transformers, scikit-learn, SciPy, and FAISS where imported. A reference environment is Python 3.12, PyTorch 2.5.1, and CUDA 12.4. GPU memory requirements depend on the dataset, batch size, and subsequence budget.

Run the commands below from the **repository root**. Examples use Bash syntax; adapt line continuations and variables when using other shells.

## 2. Training entries

| Entry | Backbone / prefix | Implementation |
|---|---|---|
| `train_drift_recent.py` | DRIFT-TIGER, [recent] | Revised partition and decoder; global subsequence fusion starts at the first SID position. |
| `train_drift_sidtier.py` | DRIFT-GLASS, [SID-tier; recent] | Original shared train_core.py implementation. |
| `DRIFT-HSTU/run_hstu_drift_c_v5.py` | DRIFT-HSTU, KuaiRec | HSTU with adaptive subsequences and structure regularization. |
| `DRIFT-HSTU/run_hstu_drift_c_v5_taobaomm.py` | DRIFT-HSTU, Taobao MM | Timestamp-free HSTU variant. |

The TIGER and GLASS entries share the objective

```text
L = L_CE + lambda_rec * L_rec + lambda_subrec * L_subrec + lambda_sub * L_sub
```

Default weights are lambda_rec=0.5, lambda_subrec=0.5, and lambda_sub=0.05; recommendation temperature is 0.07. The revised TIGER entry also corrects PAD statistics, disables Gumbel noise during evaluation, and preserves real all-zero raw SID triples. Its implementation is in `drift_method/revised.py` and `model/revised_data.py`. The two generative entries have different data and checkpoint interfaces.

## 3. Files and relative paths

```text
train_drift_recent.py       # revised DRIFT-TIGER training entry
train_drift_sidtier.py      # original DRIFT-GLASS training entry
train_core.py              # shared helpers and original training implementation
paths.py                   # repository-relative dataset and output resolution
model/                     # generative backbone and data readers
drift_method/              # partition, subsequence encoders, losses, revised model
preprocess/                # dataset preparation utilities
rqvae/                     # semantic-ID construction
scripts/                   # original single-configuration launchers
DRIFT-HSTU/                # HSTU code and its own README
data/                      # user-provided data; not distributed
outputs/                   # training artifacts created at runtime
```

`paths.py` resolves generative data paths relative to `data/` and default outputs relative to `outputs/`. Optional environment variables `DRIFT_DATA_ROOT` and `DRIFT_OUTPUT_ROOT` change those roots. Explicit `--out_dir outputs/...` examples below are relative to the current working directory, so run them from the repository root.

The examples assume this prepared-data layout:

```text
data/
  kuairec_processed/
    glass_data_256_256_256_ts/
      train_seq_len1000_sim_augmented.parquet
      valid_seq_len1000_with_history.parquet
      test_seq_len1000_with_history.parquet
  TAOBAO_MM/
    t5_data_64_128_128_seqlen1000_align/
      train_seq_len1000_sim_augmented.parquet
      valid_seq_len1000_with_history.parquet
      test_seq_len1000_with_history.parquet
```

Each row contains `history` as a list of three-level raw SID triples and `target` as one triple. KuaiRec also supplies chronological per-item `timestamps` or `timestamps_unix_s`. Histories and timestamps are truncated together to the most recent 1000 items and left-padded. Targets must remain outside the input history according to the supplied split protocol.

The revised TIGER reader adds SID-level offsets **before** adding token-zero padding. It reads only history, target, and timestamp columns; retrieval and SID-tier columns are unnecessary for this entry. A real [0,0,0] raw SID remains valid.

For revised TIGER, `--dataset_dir` names the final processed directory relative to the data root. Without that flag, the original `--data_root <parent>` convention appends `glass_data_<codebook>`, plus `_ts` when requested. Explicit split filenames can be passed with `--train_file`, `--valid_file`, and `--test_file`. Otherwise sequence-length filenames are preferred, with the original plain-filename fallback.

### Timestamp units

The KuaiRec examples below use **Unix-second** `timestamps`, including the representation produced by `preprocess/step5_inject_timestamps.py`, and therefore pass `--timestamp_unit seconds`. The revised reader centers timestamps in float64 before converting them to float32 relative days.

For separately prepared aligned-day files, use, for example, `--dataset_dir kuairec_processed/glass_data_256_256_256_aligned_days_ts --timestamp_unit days`. A fallback `timestamps_unix_s` column is always interpreted as seconds. Taobao MM examples disable timestamps with `--use_timestamps 0`.

## 4. DRIFT-TIGER: KuaiRec, L=1000, recent=50, K in {10,20,30}

This loop runs three independent experiments and writes each result to its own directory:

```bash
for K in 10 20 30; do
  python train_drift_recent.py \
    --dataset_dir kuairec_processed/glass_data_256_256_256_ts \
    --codebook 256,256,256 \
    --use_timestamps 1 --timestamp_unit seconds \
    --max_len 1000 --recent_window_r 50 \
    --K "$K" --K_min 1 --K_max "$K" \
    --batch_size 128 --infer_size 64 \
    --num_epochs 100 --early_stop 50 \
    --beam_size 20 --lr 1e-4 --weight_decay 0.01 --seed 2025 \
    --out_dir "outputs/tiger_kuairec_L1000_R50_K${K}"
done
```

For one configuration, execute the Python command with a literal value such as `--K 10 --K_min 1 --K_max 10` and the corresponding output directory.

## 5. DRIFT-TIGER: Taobao MM, L=1000, recent=50, K in {10,20,30}

The final Taobao MM directory is passed directly, matching the output layout of the Taobao MM data builder.

```bash
for K in 10 20 30; do
  python train_drift_recent.py \
    --dataset_dir TAOBAO_MM/t5_data_64_128_128_seqlen1000_align \
    --codebook 64,128,128 --use_timestamps 0 \
    --max_len 1000 --recent_window_r 50 \
    --K "$K" --K_min 1 --K_max "$K" \
    --batch_size 128 --infer_size 64 \
    --num_epochs 100 --early_stop 50 \
    --beam_size 20 --lr 1e-4 --weight_decay 0.01 --seed 2025 \
    --out_dir "outputs/tiger_taobaomm_L1000_R50_K${K}"
done
```

Here K is the number of available subsequence slots. The effective per-user count is adaptive, bounded by K_min and K_max; these examples do not force exactly K active subsequences per user. Both datasets use 1000 **items**, with the most recent 50 entering the encoder and up to 950 older items entering the subsequence branch.

### Revised TIGER outputs

Every run writes:

- `drift_recent_revised_best.pt`: model with the best validation NDCG@20.
- `drift_recent_revised_best.json`: selected epoch, validation scores, and final test scores.
- `drift_recent_revised_summary.json`: configuration, per-epoch history, and elapsed time.

The best checkpoint is restored after training, and the test split is evaluated once. Existing artifacts are not overwritten; choose a fresh output directory for a rerun. The entry runs on one device; use `--device cpu` or `--device cuda:0` to select it explicitly. Beam search retains four decoder calls and returns three SID levels.

## 6. DRIFT-GLASS examples with the original interface

The original GLASS entry accepts `--data_root`, not `--dataset_dir` or `--timestamp_unit`. Its KuaiRec data directory is constructed as `data/kuairec_processed/glass_data_256_256_256_ts/`; the timestamp convention is seconds.

```bash
for K in 10 20 30; do
  python train_drift_sidtier.py \
    --data_root kuairec_processed --codebook 256,256,256 \
    --ts_suffix --use_timestamps 1 \
    --max_len 1000 --recent_window_r 50 \
    --K "$K" --K_min 1 --K_max "$K" \
    --batch_size 128 --infer_size 64 --num_epochs 100 --early_stop 50 \
    --out_dir "outputs/glass_kuairec_L1000_R50_K${K}"
done
```

For Taobao MM, first arrange the original GLASS input files under `data/taobaomm_processed/glass_data_64_128_128/`. This original reader expects retrieval and SID-tier features from the data builder. That layout is distinct from the direct-directory TIGER example above.

```bash
for K in 10 20 30; do
  python train_drift_sidtier.py \
    --data_root taobaomm_processed --codebook 64,128,128 --use_timestamps 0 \
    --max_len 1000 --recent_window_r 50 \
    --K "$K" --K_min 1 --K_max "$K" \
    --batch_size 128 --infer_size 64 --num_epochs 100 --early_stop 50 \
    --out_dir "outputs/glass_taobaomm_L1000_R50_K${K}"
done
```

The original GLASS loop writes `drift_sidtier_best.json` and `drift_sidtier_summary.json`; it evaluates test metrics when validation improves and does not produce the revised TIGER checkpoint format. The existing `scripts/run_*.sh` files are original single-configuration launchers; the loops above explicitly enumerate all three requested budgets.

## 7. DRIFT-HSTU and preprocessing

See [DRIFT-HSTU/README.md](DRIFT-HSTU/README.md) for its input formats, Stage-0 preparation, and KuaiRec/Taobao MM commands at L=1000, recent_r=50, and K_max in {10,20,30}. HSTU scores real item IDs rather than generating SID triples. Its fixed-K study runners use separate data contracts and are documented separately from the adaptive v5 entries.

The preprocessing utilities cover raw interactions, item text embeddings, RQ-VAE semantic IDs, generation-data construction, and optional timestamp attachment. They do not ship datasets or checkpoints. Configure the relative locations expected by each utility and prepare the sequence-length-1000 splits before running the examples. Supplied splits are consumed as provided; dataset-specific builders and archived studies may use different holdout protocols.

The code and examples use relative locations and contain no machine-specific installation instructions. Resolve all data dependencies before starting a training run.

