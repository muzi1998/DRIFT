#!/usr/bin/env python3
"""
prepare_glass_taobaomm_seq1000_align.py
==============================================================================
Based on prepare_glass_taobaomm_seq1000.py, with an added user-alignment step:
  read the user list from t5_data_64_128_128 (the original version with
  MAX_SEQ_LEN=50, MIN_SEQ_LEN=40) and keep only those users during the seq1000
  processing, so that the two output directories contain exactly the same user
  set.

Key changes (relative to prepare_glass_taobaomm_seq1000.py):
  1. Added ALIGN_USER_SOURCE: points at the output directory of the original version
  2. Added load_align_users(): read the user set from the original version's
     train_sim_augmented.parquet
  3. _process_dataframe_rows gained an allow_users filter
  4. step1_build_user_index gained an allow_users filter
  5. OUTPUT_DATA_DIR changed to t5_data_64_128_128_seqlen1000_align

No existing script file is modified, overwritten or deleted.
==============================================================================
"""
import pandas as pd
import numpy as np
import os
import json
import glob
import gc
from tqdm import tqdm

# =============================================================================
# Global configuration parameters
# =============================================================================

BASE_DIR = "./TAOBAO_MM"
CODEBOOK = [64, 128, 128]
MAX_SEQ_LEN = 1000
MIN_SEQ_LEN = 5
MAX_RETRIEVAL_LEN = 30

TRAIN_SHARD_DIR = os.path.join(BASE_DIR, "micro/train")
TEST_SHARD_DIR = os.path.join(BASE_DIR, "micro/test")
ITEM_EMB_PATH = os.path.join(BASE_DIR, "rqvae_data/item_emb.parquet")
RQVAE_CODES_PATH = os.path.join(BASE_DIR, f'TAOBAO_MM_t5_rqvae_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy')

CODEBOOK_VEC_PATH = os.path.join(BASE_DIR, f'codebook_vectors_layer1_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy')
CODEBOOK_SIM_PATH = os.path.join(BASE_DIR, f"codebook_similarity_matrix_{CODEBOOK[0]}.npy")

# Output directory name (carries the _align suffix so that it does not overwrite
# the existing directory)
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, f"t5_data_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}_seqlen1000_align")
USER_INDEX_PATH = os.path.join(BASE_DIR, f"user_history_groups_simsid_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}_seqlen1000_align_codes.json")

# Output directory of the original version (used to read the alignment user set)
ALIGN_USER_SOURCE = os.path.join(BASE_DIR, f"t5_data_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}")

BIN_BOUNDARIES = np.array([-1.0, -0.1, 0.0, 0.1, 1.0])

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


# =============================================================================
# Loading the alignment user set
# =============================================================================

def load_align_users():
    """Read the user set from the original version's train_sim_augmented.parquet
    in t5_data_64_128_128.

    The original version (MAX_SEQ_LEN=50, MIN_SEQ_LEN=40) retained 34,535 users.
    This function returns those users as a set of strings, used to filter the
    later processing stages.
    """
    align_train = os.path.join(ALIGN_USER_SOURCE, "train_sim_augmented.parquet")
    if not os.path.exists(align_train):
        print(f"[warn] the alignment source file does not exist: {align_train}")
        print(f"     no user filtering will be applied (equivalent to the original seq1000 behavior)")
        return None

    df = pd.read_parquet(align_train, columns=['user'])
    users = set(df['user'].astype(str).unique())
    print(f"[Align] loaded the alignment user set from {ALIGN_USER_SOURCE}: {len(users)} users")
    del df
    gc.collect()
    return users


# =============================================================================
# Utility functions: artifact generation and loading (identical to the seq1000 version)
# =============================================================================

def ensure_codebook_artifacts():
    if os.path.exists(CODEBOOK_VEC_PATH) and os.path.exists(CODEBOOK_SIM_PATH):
        print("[done] [Artifacts] codebook vectors and similarity matrix already exist, skipping generation.")
        return
    print("\n[Artifacts] Generating the codebook vectors and similarity matrix ...")
    if not os.path.exists(ITEM_EMB_PATH):
        raise FileNotFoundError(f"item embedding file missing: {ITEM_EMB_PATH}")
    if not os.path.exists(RQVAE_CODES_PATH):
        raise FileNotFoundError(f"RQ-VAE code file missing: {RQVAE_CODES_PATH}")
    item_df = pd.read_parquet(ITEM_EMB_PATH)
    rqvae_codes = np.load(RQVAE_CODES_PATH)
    item_embeddings = np.vstack(item_df['embedding'].values).astype(np.float32)
    emb_dim = item_embeddings.shape[1]
    codebook_vec = np.zeros((CODEBOOK[0], emb_dim), dtype=np.float32)
    codebook_cnt = np.zeros(CODEBOOK[0], dtype=np.int32)
    for i in tqdm(range(len(item_embeddings)), desc="   aggregating"):
        layer1_code = rqvae_codes[i, 0]
        codebook_vec[layer1_code] += item_embeddings[i]
        codebook_cnt[layer1_code] += 1
    global_mean = np.mean(item_embeddings, axis=0)
    for code in range(CODEBOOK[0]):
        if codebook_cnt[code] > 0:
            codebook_vec[code] /= codebook_cnt[code]
        else:
            codebook_vec[code] = global_mean
    np.save(CODEBOOK_VEC_PATH, codebook_vec)
    norms = np.linalg.norm(codebook_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed_vec = codebook_vec / norms
    sim_matrix = np.dot(normed_vec, normed_vec.T)
    np.save(CODEBOOK_SIM_PATH, sim_matrix)
    del item_df, item_embeddings, rqvae_codes, codebook_vec, sim_matrix
    gc.collect()


def load_basic_resources():
    print("[Resource] Loading the basic resources ...")
    item_df = pd.read_parquet(ITEM_EMB_PATH, columns=['item_id'])
    hash_to_index = {uid: idx for idx, uid in enumerate(item_df['item_id'])}
    print(f"   total number of items: {len(hash_to_index)}")
    if not os.path.exists(RQVAE_CODES_PATH):
        semantic_ids = np.random.randint(0, 256, size=(len(item_df), 3)).astype(np.int16)
    else:
        semantic_ids = np.load(RQVAE_CODES_PATH)
        print(f"   semantic code shape: {semantic_ids.shape}")
    return hash_to_index, semantic_ids


# =============================================================================
# Core data processing steps
# =============================================================================

def step1_build_user_index(hash_to_index, semantic_ids, allow_users=None):
    """Build the user history group index, keeping only the users in allow_users."""
    print("\nStep 1: building the user history group index (keyed by the level-1 semantic code)")
    if allow_users is not None:
        print(f"   [info] user alignment mode: keeping only the {len(allow_users)} aligned users")

    train_files = sorted(glob.glob(os.path.join(TRAIN_SHARD_DIR, "*.parquet")))
    test_files = sorted(glob.glob(os.path.join(TEST_SHARD_DIR, "*.parquet")))
    all_files = train_files + test_files
    if not all_files:
        return
    user_groups = {}
    skipped_not_in_align = 0
    for file_path in tqdm(all_files, desc="   processing shards"):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"   failed to read {file_path}: {e}")
            continue
        if '129_1' not in df.columns or '150_2_180' not in df.columns:
            continue
        users = df['129_1'].values
        sequences = df['150_2_180'].values
        for user_raw, seq_raw in zip(users, sequences):
            user_id = str(user_raw)
            # user alignment filter
            if allow_users is not None and user_id not in allow_users:
                skipped_not_in_align += 1
                continue
            if isinstance(seq_raw, np.ndarray):
                seq_raw = seq_raw.tolist()
            if not isinstance(seq_raw, list):
                continue
            history = seq_raw[:-50]
            if not history:
                continue
            for item_hash in history:
                idx = hash_to_index.get(item_hash)
                if idx is None:
                    continue
                codes = semantic_ids[idx].tolist()
                l1 = int(codes[0])
                if l1 not in user_groups.setdefault(user_id, {}):
                    user_groups[user_id][l1] = []
                user_groups[user_id][l1].append(codes)
        del df
        gc.collect()
    print(f"   index built, {len(user_groups)} users in total.")
    if allow_users is not None:
        print(f"   skipped (not in the alignment user set): {skipped_not_in_align}")
    with open(USER_INDEX_PATH, 'w') as f:
        json.dump(user_groups, f)
    del user_groups
    gc.collect()


def _process_dataframe_rows(df, hash_to_index, semantic_ids, allow_users=None):
    """Process a single DataFrame, keeping only the users in allow_users."""
    processed = []
    stats = {'missing_target': 0, 'short_history': 0, 'insufficient_sequence': 0, 'not_in_align': 0}
    users = df['129_1'].values
    sequences = df['150_2_180'].values
    targets = df['205'].values
    for user_raw, seq_raw, target_raw in zip(users, sequences, targets):
        user_id = str(user_raw)
        # user alignment filter
        if allow_users is not None and user_id not in allow_users:
            stats['not_in_align'] += 1
            continue
        tgt_idx = hash_to_index.get(target_raw)
        if tgt_idx is None:
            stats['missing_target'] += 1
            continue
        target_codes = semantic_ids[tgt_idx].tolist()
        if isinstance(seq_raw, np.ndarray):
            seq_raw = seq_raw.tolist()
        if not isinstance(seq_raw, list):
            seq_raw = []
        valid_sequence = []
        for item_hash in seq_raw:
            idx = hash_to_index.get(item_hash)
            if idx is not None:
                valid_sequence.append(semantic_ids[idx].tolist())
        if len(valid_sequence) < MIN_SEQ_LEN + 2:
            stats['insufficient_sequence'] += 1
            continue
        test_target = valid_sequence[-1]
        valid_target = valid_sequence[-2]
        train_history = valid_sequence[:-2]
        train_history = train_history[-MAX_SEQ_LEN:]
        if len(train_history) < MIN_SEQ_LEN:
            stats['short_history'] += 1
            continue
        processed.append({'user': user_id, 'history': train_history, 'target': test_target, 'dataset_type': 'test'})
        processed.append({'user': user_id, 'history': train_history, 'target': valid_target, 'dataset_type': 'valid'})
        processed.append({'user': user_id, 'history': train_history[:-1], 'target': train_history[-1], 'dataset_type': 'train'})
    return processed, stats


def _process_folder_to_parquet(folder_path, file_pattern, hash_to_index, semantic_ids, out_filename, allow_users=None):
    files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    if not files:
        return
    print(f"   [info] building {out_filename} ...")
    all_data = []
    total_stats = {'missing_target': 0, 'short_history': 0, 'insufficient_sequence': 0, 'not_in_align': 0}
    for f in tqdm(files, desc="   reading shards"):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"   [warn] failed to read {f}: {e}")
            continue
        rows, stats = _process_dataframe_rows(df, hash_to_index, semantic_ids, allow_users)
        all_data.extend(rows)
        for k in total_stats:
            total_stats[k] += stats[k]
        del df
        gc.collect()
    if not all_data:
        return
    final_df = pd.DataFrame(all_data)
    out_path = os.path.join(OUTPUT_DATA_DIR, out_filename)
    final_df.to_parquet(out_path)
    print(f"   [done] saved: {out_path} (samples: {len(final_df)})")
    print(f"      dropped (missing target):     {total_stats['missing_target']}")
    print(f"      dropped (history too short):  {total_stats['short_history']}")
    print(f"      dropped (sequence too short): {total_stats['insufficient_sequence']}")
    if allow_users is not None:
        print(f"      dropped (user not in the aligned set): {total_stats['not_in_align']}")
    del all_data, final_df
    gc.collect()


def step2_create_basic_datasets(hash_to_index, semantic_ids, allow_users=None):
    print(f"\nStep 2: building the base dataset "
          f"(MAX_SEQ={MAX_SEQ_LEN}, MIN_SEQ={MIN_SEQ_LEN})")
    if allow_users is not None:
        print(f"   [info] user-alignment mode: keeping only the "
              f"{len(allow_users)} aligned users")
    _process_folder_to_parquet(TRAIN_SHARD_DIR, "train-shard-*.parquet", hash_to_index, semantic_ids, "all_samples.parquet", allow_users)
    _process_folder_to_parquet(TEST_SHARD_DIR, "test-shard-*.parquet", hash_to_index, semantic_ids, "all_samples.parquet", allow_users)
    all_samples_path = os.path.join(OUTPUT_DATA_DIR, "all_samples.parquet")
    if os.path.exists(all_samples_path):
        df_all = pd.read_parquet(all_samples_path)
        for dt, name in [('train', 'train.parquet'), ('valid', 'valid.parquet'), ('test', 'test.parquet')]:
            sub = df_all[df_all['dataset_type'] == dt]
            sub.to_parquet(os.path.join(OUTPUT_DATA_DIR, name), index=False)
            print(f"   [done] {name}: {len(sub)} rows")
        os.remove(all_samples_path)


def step3_augment_train_data():
    print("\nStep 3: retrieval augmentation for the training split")
    train_file = os.path.join(OUTPUT_DATA_DIR, "train.parquet")
    if not os.path.exists(train_file):
        return
    if not os.path.exists(USER_INDEX_PATH):
        return
    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)
    df = pd.read_parquet(train_file)
    retrieved_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   retrieving history"):
        user_id = str(row['user'])
        target = row['target']
        l1 = str(target[0]) if target is not None and len(target) > 0 else None
        if l1 and user_id in user_index:
            full = user_index[user_id].get(l1, [])
            retrieved = full[-MAX_RETRIEVAL_LEN:]
        else:
            retrieved = []
        retrieved_list.append(retrieved)
    df['same_sid_history'] = retrieved_list
    df.to_parquet(os.path.join(OUTPUT_DATA_DIR, "train_sim_augmented.parquet"))
    del df, user_index, retrieved_list
    gc.collect()


def step3_split_test_with_histories():
    print("\nStep 3b: retrieval augmentation for the valid/test splits")
    if not os.path.exists(USER_INDEX_PATH):
        return
    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)
    for split_name, out_name in [('valid.parquet', 'valid_with_history.parquet'), ('test.parquet', 'test_with_history.parquet')]:
        fpath = os.path.join(OUTPUT_DATA_DIR, split_name)
        if not os.path.exists(fpath):
            continue
        df = pd.read_parquet(fpath)
        users = df['user'].astype(str).tolist()
        new_cols = {}
        for sid in tqdm(range(CODEBOOK[0]), desc=f"   {split_name}"):
            col_data = []
            for uid in users:
                hist = user_index.get(uid, {}).get(str(sid), [])
                col_data.append(hist[-MAX_RETRIEVAL_LEN:])
            new_cols[f'sid_{sid}_history'] = col_data
        df_aug = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        out_path = os.path.join(OUTPUT_DATA_DIR, out_name)
        df_aug.to_parquet(out_path, index=False)
        print(f"   [done] {out_name}: {len(df_aug)} rows")
        del df, df_aug, new_cols
        gc.collect()


# =============================================================================
# Main pipeline
# =============================================================================

if __name__ == "__main__":
    print(f"[start] building the aligned seq1000 dataset "
          f"(MAX_SEQ={MAX_SEQ_LEN}, MIN_SEQ={MIN_SEQ_LEN})")
    print(f"   output directory: {OUTPUT_DATA_DIR}")
    print(f"   alignment source: {ALIGN_USER_SOURCE}")

    # Load the user set to align with, so both dataset variants cover the same users.
    align_users = load_align_users()

    ensure_codebook_artifacts()
    hash2idx, semantic_ids = load_basic_resources()

    # Step 1: build the user index (aligned users only)
    step1_build_user_index(hash2idx, semantic_ids, allow_users=align_users)

    # Step 2: build the base dataset (aligned users only)
    step2_create_basic_datasets(hash2idx, semantic_ids, allow_users=align_users)

    del hash2idx, semantic_ids
    gc.collect()

    # Step 3: retrieval augmentation
    step3_augment_train_data()
    step3_split_test_with_histories()

    # Remove the intermediate files
    for f in ["train.parquet", "valid.parquet", "test.parquet"]:
        p = os.path.join(OUTPUT_DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    # Verify that the user sets really match
    print("\n[check] verifying the user alignment ...")
    align_train = os.path.join(OUTPUT_DATA_DIR, "train_sim_augmented.parquet")
    orig_train = os.path.join(ALIGN_USER_SOURCE, "train_sim_augmented.parquet")
    if os.path.exists(align_train) and os.path.exists(orig_train):
        df_new = pd.read_parquet(align_train, columns=['user'])
        df_orig = pd.read_parquet(orig_train, columns=['user'])
        u_new = set(df_new['user'].astype(str).unique())
        u_orig = set(df_orig['user'].astype(str).unique())
        print(f"   users in this build:   {len(u_new)}")
        print(f"   users in the reference: {len(u_orig)}")
        print(f"   user sets identical:    {u_new == u_orig}")
        if u_new != u_orig:
            print(f"   only in this build:   {len(u_new - u_orig)}")
            print(f"   only in the reference: {len(u_orig - u_new)}")

    print("\n[done] aligned seq1000 dataset built successfully.")
    print(f"   output directory: {OUTPUT_DATA_DIR}")
