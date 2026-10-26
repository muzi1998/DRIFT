import pandas as pd
import numpy as np
import os
import json
import gc
import ast
from tqdm import tqdm
import argparse

# =============================================================================
# Common utility functions
# =============================================================================

def _parse_embedding_cell(cell):
    """Parse a single embedding cell into a 1D float32 ndarray.

    Supports the following storage forms:
      1) list / tuple: converted directly to an ndarray
      2) np.ndarray: used as is
      3) str: shaped like "[0.1, 0.2, ...]" or "0.1 0.2 ...", parsed safely
    """
    if isinstance(cell, np.ndarray):
        return cell.astype(np.float32, copy=False)
    if isinstance(cell, (list, tuple)):
        return np.asarray(cell, dtype=np.float32)
    if isinstance(cell, (bytes, bytearray)):
        cell = cell.decode('utf-8', errors='ignore')
    if isinstance(cell, str):
        s = cell.strip()
        # Try ast.literal_eval first (it handles the "[a, b, c]" form)
        try:
            parsed = ast.literal_eval(s)
            return np.asarray(parsed, dtype=np.float32)
        except (ValueError, SyntaxError):
            pass
        # Fall back to splitting on whitespace/commas
        s_clean = s.strip('[]() \n\t').replace(',', ' ')
        parts = [p for p in s_clean.split() if p]
        return np.asarray(parts, dtype=np.float32)
    # Any other unknown type: attempt a direct conversion
    return np.asarray(cell, dtype=np.float32)


def load_item_embeddings(item_df, col='embedding'):
    """Read the embedding column of the item DataFrame and return an (N, D)
    float32 matrix.

    All of list/ndarray/str storage forms are handled uniformly, which avoids the
    ValueError that a plain np.vstack+astype would raise when it encounters a
    string cell.
    """
    values = item_df[col].values
    first = values[0]
    # Fast path: if the first element is already a stackable numeric array/list,
    # try a single vstack.
    if isinstance(first, np.ndarray) or isinstance(first, (list, tuple)):
        try:
            return np.vstack(values).astype(np.float32, copy=False)
        except (ValueError, TypeError):
            pass  # fall back to row-by-row parsing

    # Row-by-row parsing (handles mixed types / strings)
    parsed = [_parse_embedding_cell(v) for v in values]
    return np.vstack(parsed).astype(np.float32, copy=False)

# =============================================================================
# Global configuration parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_len",
    type=int,
    default="50"
)
args = parser.parse_args()

BASE_DIR = "./kuairec_processed"
CODEBOOK = [256, 256, 256]
MAX_SEQ_LEN = args.max_len # 500 50
MIN_SEQ_LEN = 5
MAX_RETRIEVAL_LEN = 30

ITEM_EMB_PATH = os.path.join(BASE_DIR, "item_embeddings.parquet")
RQVAE_CODES_PATH = os.path.join(BASE_DIR, f"kuairec_rqvae_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy")

TRAIN_INTER_PATH = os.path.join(BASE_DIR, f"train_seq_len{MAX_SEQ_LEN}.parquet")
VALID_INTER_PATH = os.path.join(BASE_DIR, f"valid_seq_len{MAX_SEQ_LEN}.parquet")
TEST_INTER_PATH  = os.path.join(BASE_DIR, f"test_seq_len{MAX_SEQ_LEN}.parquet")
MERGED_ALL_PATH  = os.path.join(BASE_DIR, "merged_all.parquet")

CODEBOOK_VEC_PATH = os.path.join(BASE_DIR, f"codebook_vectors_layer1_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy")
CODEBOOK_SIM_PATH = os.path.join(BASE_DIR, f"codebook_similarity_matrix_{CODEBOOK[0]}.npy")

OUTPUT_DATA_DIR = os.path.join(BASE_DIR, f"glass_data_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}")
USER_INDEX_PATH = os.path.join(BASE_DIR, f"user_history_groups_simsid_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}_codes.json")

BIN_BOUNDARIES = np.array([-1.0, -0.1, 0.0, 0.1, 1.0])

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


def ensure_codebook_artifacts():
    if os.path.exists(CODEBOOK_VEC_PATH) and os.path.exists(CODEBOOK_SIM_PATH):
        print("[done] [Artifacts] codebook vectors and similarity matrix already exist, skipping generation.")
        return

    print("\n" + "=" * 60)
    print("[Artifacts] Generating the codebook vectors and similarity matrix ...")
    print("=" * 60)

    item_df = pd.read_parquet(ITEM_EMB_PATH)
    rqvae_codes = np.load(RQVAE_CODES_PATH)
    item_embeddings = load_item_embeddings(item_df, col='embedding')
    emb_dim = item_embeddings.shape[1]

    print(f"   Aggregating the level-1 codebook vectors ({CODEBOOK[0]} codes in total) ...")
    codebook_vec = np.zeros((CODEBOOK[0], emb_dim), dtype=np.float32)
    codebook_cnt = np.zeros(CODEBOOK[0], dtype=np.int32)

    for i in tqdm(range(len(item_embeddings)), desc="   aggregating"):
        layer1_code = int(rqvae_codes[i, 0])
        codebook_vec[layer1_code] += item_embeddings[i]
        codebook_cnt[layer1_code] += 1

    global_mean = np.mean(item_embeddings, axis=0)
    for code in range(CODEBOOK[0]):
        if codebook_cnt[code] > 0:
            codebook_vec[code] /= codebook_cnt[code]
        else:
            codebook_vec[code] = global_mean

    np.save(CODEBOOK_VEC_PATH, codebook_vec)
    print(f"   [done] codebook vectors saved: {CODEBOOK_VEC_PATH}")

    norms = np.linalg.norm(codebook_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed_vec = codebook_vec / norms
    sim_matrix = np.dot(normed_vec, normed_vec.T)
    np.save(CODEBOOK_SIM_PATH, sim_matrix)
    print(f"   [done] similarity matrix saved: {CODEBOOK_SIM_PATH}")

    del item_df, item_embeddings, rqvae_codes, codebook_vec, sim_matrix
    gc.collect()
    print("[done] [Artifacts] generation complete!\n")


def load_basic_resources():
    print("[Resource] Loading the basic resources ...")
    item_df = pd.read_parquet(ITEM_EMB_PATH, columns=['video_id'])
    video_id_to_index = {int(vid): idx for idx, vid in enumerate(item_df['video_id'])}
    print(f"   total number of items: {len(video_id_to_index)}")

    rqvae_codes = np.load(RQVAE_CODES_PATH)
    semantic_ids = rqvae_codes[:, :3].astype(np.int32)
    print(f"   semantic code shape: {semantic_ids.shape}")

    return video_id_to_index, semantic_ids


def build_user_sequences_fast(video_id_to_index, semantic_ids):
    """
    Efficient version: build the user sequences with vectorized operations,
    avoiding a row-by-row iteration over groupby.
    """
    print("\n" + "=" * 60)
    print("[Preprocessing] Building the user interaction sequences (vectorized, sorted by time) ...")
    print("=" * 60)

    if os.path.exists(MERGED_ALL_PATH):
        print(f"   Reading {MERGED_ALL_PATH} ...")
        df_all = pd.read_parquet(MERGED_ALL_PATH, columns=['user_id', 'video_id', 'timestamp'])
    else:
        dfs = []
        for path in [TRAIN_INTER_PATH, VALID_INTER_PATH, TEST_INTER_PATH]:
            if os.path.exists(path):
                dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
        df_all = pd.concat(dfs, ignore_index=True)

    print(f"   total interactions: {len(df_all)}")

    # Vectorized mapping video_id -> item_index
    print("   Vectorized mapping video_id -> item_index ...")
    vid_array = df_all['video_id'].values.astype(np.int64)
    # Build the lookup array (sized by the maximum video_id)
    max_vid = int(vid_array.max()) + 1
    vid_map = np.full(max_vid, -1, dtype=np.int32)
    for vid, idx in video_id_to_index.items():
        if vid < max_vid:
            vid_map[vid] = idx

    item_indices = vid_map[vid_array]
    valid_mask = item_indices >= 0

    # Drop the invalid items
    df_valid = df_all[valid_mask].copy()
    df_valid['item_idx'] = item_indices[valid_mask]
    del df_all, item_indices, valid_mask
    gc.collect()

    print(f"   valid interactions: {len(df_valid)}")

    # Sort by user_id, timestamp
    print("   Sorting ...")
    df_valid = df_valid.sort_values(['user_id', 'timestamp'])

    # Vectorized construction of the semantic codes
    print("   Vectorized construction of the semantic codes ...")
    item_idx_arr = df_valid['item_idx'].values
    user_id_arr  = df_valid['user_id'].values

    # Fetch the semantic code (3 dimensions) of every interaction
    codes_arr = semantic_ids[item_idx_arr]  # (N, 3)

    # Group by user (exploiting the contiguity created by the sort)
    print("   Grouping by user ...")
    user_ids_unique, user_start_idx, user_counts = np.unique(
        user_id_arr, return_index=True, return_counts=True
    )

    print(f"   total number of users: {len(user_ids_unique)}")

    user_sequences = {}
    for i in tqdm(range(len(user_ids_unique)), desc="   building user sequences"):
        uid = int(user_ids_unique[i])
        start = user_start_idx[i]
        count = user_counts[i]
        user_codes = codes_arr[start:start+count].tolist()  # list of [c1,c2,c3]
        user_sequences[uid] = user_codes

    del df_valid, codes_arr, user_id_arr, item_idx_arr
    gc.collect()
    print(f"   [done] user sequences built, {len(user_sequences)} users in total.")
    return user_sequences


def step1_build_user_index(user_sequences):
    print("\n" + "=" * 60)
    print("Step 1: building the user history group index (keyed by the level-1 semantic code)")
    print("=" * 60)

    user_groups = {}

    for user_id, codes_list in tqdm(user_sequences.items(), desc="   building the index"):
        history = codes_list[:-2] if len(codes_list) > 2 else []
        if not history:
            continue
        uid_str = str(user_id)
        for codes in history:
            l1 = str(int(codes[0]))
            if uid_str not in user_groups:
                user_groups[uid_str] = {}
            if l1 not in user_groups[uid_str]:
                user_groups[uid_str][l1] = []
            user_groups[uid_str][l1].append(codes)

    print(f"   index built, {len(user_groups)} users in total.")
    with open(USER_INDEX_PATH, 'w') as f:
        json.dump(user_groups, f)
    print(f"   [done] index saved: {USER_INDEX_PATH}")

    del user_groups
    gc.collect()


def step2_create_basic_datasets(user_sequences):
    print("\n" + "=" * 60)
    print(f"Step 2: generating the basic datasets (leave-last-out strategy, hard truncation at {MAX_SEQ_LEN}, minimum valid length {MIN_SEQ_LEN})")
    print("=" * 60)

    train_data, valid_data, test_data = [], [], []
    skip_short = 0

    for user_id, codes_list in tqdm(user_sequences.items(), desc="   generating samples"):
        if len(codes_list) < MIN_SEQ_LEN + 2:
            skip_short += 1
            continue

        test_target  = codes_list[-1]
        valid_target = codes_list[-2]
        history_all  = codes_list[:-2]
        history = history_all[-MAX_SEQ_LEN:]

        if len(history) < MIN_SEQ_LEN:
            skip_short += 1
            continue

        test_data.append({'user': str(user_id), 'history': history, 'target': test_target})
        valid_data.append({'user': str(user_id), 'history': history, 'target': valid_target})
        if len(history) >= 2:
            train_data.append({'user': str(user_id), 'history': history[:-1], 'target': history[-1]})

    print(f"   skipped (sequence too short): {skip_short}")
    print(f"   training set: {len(train_data)} samples")
    print(f"   validation set: {len(valid_data)} samples")
    print(f"   test set: {len(test_data)} samples")

    for name, data in [(f'train_seq_len{MAX_SEQ_LEN}', train_data), (f'valid_seq_len{MAX_SEQ_LEN}', valid_data), (f'test_seq_len{MAX_SEQ_LEN}', test_data)]:
        df = pd.DataFrame(data)
        path = os.path.join(OUTPUT_DATA_DIR, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"   {name}.parquet saved: {path}")
        del df
        gc.collect()

    print("[done] Step 2 complete!\n")


def step3_augment_train_data():
    print("\n" + "=" * 60)
    print("Step 3: retrieval augmentation of the training data (based on the user index)")
    print("=" * 60)

    train_file = os.path.join(OUTPUT_DATA_DIR, f"train_seq_len{MAX_SEQ_LEN}.parquet")
    if not os.path.exists(train_file):
        print(f"[warn] {train_file} not found, skipping Step 3.")
        return

    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)

    df = pd.read_parquet(train_file)
    retrieved_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="   retrieving history"):
        user_id = str(row['user'])
        target = row['target']
        l1 = str(int(target[0])) if target is not None and len(target) > 0 else None

        if l1 and user_id in user_index:
            full = user_index[user_id].get(l1, [])
            retrieved = full[-MAX_RETRIEVAL_LEN:]
        else:
            retrieved = []
        retrieved_list.append(retrieved)

    df['same_sid_history'] = retrieved_list
    out_path = os.path.join(OUTPUT_DATA_DIR, f"train_seq_len{MAX_SEQ_LEN}_sim_augmented.parquet")
    df.to_parquet(out_path, index=False)
    print(f"   augmented training data saved: {out_path}")

    del df, user_index, retrieved_list
    gc.collect()
    print("[done] Step 3 complete!\n")


def step3_split_test_with_histories():
    print("\n" + "=" * 60)
    print("Step 3b: adding the retrieved histories to the validation and test sets")
    print("=" * 60)

    with open(USER_INDEX_PATH, 'r') as f:
        user_index = json.load(f)

    for split_name, out_name in [(f'valid_seq_len{MAX_SEQ_LEN}', f'valid_seq_len{MAX_SEQ_LEN}_with_history'), (f'test_seq_len{MAX_SEQ_LEN}', f'test_seq_len{MAX_SEQ_LEN}_with_history')]:
        split_file = os.path.join(OUTPUT_DATA_DIR, f"{split_name}.parquet")
        if not os.path.exists(split_file):
            print(f"[warn] {split_file} not found")
            continue

        df = pd.read_parquet(split_file)
        users = df['user'].astype(str).tolist()

        new_cols = {}
        for sid in tqdm(range(CODEBOOK[0]), desc=f"   building the per-code history columns for {split_name}"):
            sid_str = str(sid)
            col_data = []
            for uid in users:
                hist = user_index.get(uid, {}).get(sid_str, [])
                col_data.append(hist[-MAX_RETRIEVAL_LEN:])
            new_cols[f'sid_{sid}_history'] = col_data

        df_aug = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        out_path = os.path.join(OUTPUT_DATA_DIR, f"{out_name}.parquet")
        df_aug.to_parquet(out_path, index=False)
        print(f"   [done] {split_name} split: {len(df_aug)} samples, saved to: {out_path}")
        del df, df_aug, new_cols
        gc.collect()

    print("[done] Step 3b complete!\n")


def step4_add_sidtier_and_cleanup():
    print("\n" + "=" * 60)
    print("Step 4: adding the SIDTier features and cleaning up the intermediate files")
    print("=" * 60)

    print("   Loading the item embeddings and codebook vectors ...")
    item_df = pd.read_parquet(ITEM_EMB_PATH)
    item_embeddings = load_item_embeddings(item_df, col='embedding')
    video_id_to_idx = {int(vid): i for i, vid in enumerate(item_df['video_id'])}
    codebook_vec = np.load(CODEBOOK_VEC_PATH).astype(np.float32)
    print(f"   item embedding shape: {item_embeddings.shape}, codebook shape: {codebook_vec.shape}")

    # Build the users' long-term histories (vectorized)
    print("   Building the users' long-term histories (item indices) ...")
    if os.path.exists(MERGED_ALL_PATH):
        df_all = pd.read_parquet(MERGED_ALL_PATH, columns=['user_id', 'video_id', 'timestamp'])
    else:
        dfs = []
        for path in [TRAIN_INTER_PATH, VALID_INTER_PATH, TEST_INTER_PATH]:
            if os.path.exists(path):
                dfs.append(pd.read_parquet(path, columns=['user_id', 'video_id', 'timestamp']))
        df_all = pd.concat(dfs, ignore_index=True)

    df_all = df_all.sort_values(['user_id', 'timestamp'])

    vid_array = df_all['video_id'].values.astype(np.int64)
    max_vid = int(vid_array.max()) + 1
    vid_map = np.full(max_vid, -1, dtype=np.int32)
    for vid, idx in video_id_to_idx.items():
        if vid < max_vid:
            vid_map[vid] = idx
    item_idx_arr = vid_map[vid_array]
    valid_mask = item_idx_arr >= 0

    df_valid = df_all[valid_mask].copy()
    df_valid['item_idx'] = item_idx_arr[valid_mask]
    del df_all, item_idx_arr, valid_mask
    gc.collect()

    user_id_arr = df_valid['user_id'].values
    item_idx_arr2 = df_valid['item_idx'].values
    user_ids_unique, user_start_idx, user_counts = np.unique(user_id_arr, return_index=True, return_counts=True)

    user_long_hist = {}
    for i in tqdm(range(len(user_ids_unique)), desc="   building the user histories"):
        uid = str(int(user_ids_unique[i]))
        start = user_start_idx[i]
        count = user_counts[i]
        indices = item_idx_arr2[start:start+count].tolist()
        # Drop the last two interactions, since they are the validation and test
        # targets under leave-one-out and must not leak into the history.
        if len(indices) > 2:
            user_long_hist[uid] = indices[:-2]
        elif indices:
            user_long_hist[uid] = indices

    print(f"   users' long-term histories built, {len(user_long_hist)} users in total.")
    del df_valid, user_id_arr, item_idx_arr2
    gc.collect()

    def compute_sidtier_batch(history_indices_list):
        features_list = []
        for hist_idx in history_indices_list:
            if len(hist_idx) == 0:
                features_list.append(np.zeros(CODEBOOK[0] * 4, dtype=np.float32))
                continue
            hist_embs = item_embeddings[hist_idx]
            sims = np.dot(hist_embs, codebook_vec.T)
            sidtier = np.zeros(CODEBOOK[0] * 4, dtype=np.float32)
            for c in range(CODEBOOK[0]):
                col_sims = sims[:, c]
                counts, _ = np.histogram(col_sims, bins=BIN_BOUNDARIES)
                total = len(col_sims)
                norm_counts = counts / total if total > 0 else np.zeros(4)
                sidtier[c*4:(c+1)*4] = norm_counts
            features_list.append(sidtier)
        return features_list

    target_files = [
        (f"valid_seq_len{MAX_SEQ_LEN}_with_history.parquet", "validation set"),
        (f"test_seq_len{MAX_SEQ_LEN}_with_history.parquet",  "test set"),
        (f"train_seq_len{MAX_SEQ_LEN}_sim_augmented.parquet","augmented training set"),
    ]

    batch_size = 500
    for fname, desc in target_files:
        fpath = os.path.join(OUTPUT_DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   [warn] file {fname} does not exist, skipping.")
            continue

        print(f"\n   Processing the {desc}: {fname}")
        df = pd.read_parquet(fpath)
        users = df['user'].astype(str).tolist()
        hist_indices = [user_long_hist.get(uid, []) for uid in users]

        all_sidtier = []
        for i in tqdm(range(0, len(users), batch_size), desc=f"   computing sidtier"):
            batch = hist_indices[i:i+batch_size]
            batch_feat = compute_sidtier_batch(batch)
            all_sidtier.extend(batch_feat)

        df['sidtier'] = all_sidtier
        df.to_parquet(fpath, index=False)
        print(f"     sidtier features updated and saved.")
        del df
        gc.collect()

    print("\n   Cleaning up the intermediate files ...")
    for f in [f"train_seq_len{MAX_SEQ_LEN}.parquet", f"valid_seq_len{MAX_SEQ_LEN}.parquet", f"test_seq_len{MAX_SEQ_LEN}.parquet"]:
        p = os.path.join(OUTPUT_DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"     removed {f}")

    del item_df, item_embeddings, codebook_vec, user_long_hist
    gc.collect()
    print("[done] Step 4 complete!")


if __name__ == "__main__":
    print(f"[start] KuaiRec data preprocessing begins (optimized version)")
    print(f"   config: Codebook={CODEBOOK}, MAX_SEQ={MAX_SEQ_LEN}, MIN_SEQ={MIN_SEQ_LEN}")
    print(f"   output directory: {OUTPUT_DATA_DIR}")

    ensure_codebook_artifacts()
    video_id_to_index, semantic_ids = load_basic_resources()
    user_sequences = build_user_sequences_fast(video_id_to_index, semantic_ids)
    step1_build_user_index(user_sequences)
    step2_create_basic_datasets(user_sequences)

    del video_id_to_index, semantic_ids, user_sequences
    gc.collect()

    step3_augment_train_data()
    step3_split_test_with_histories()
    step4_add_sidtier_and_cleanup()

    print("\n[done] all steps finished!")
    print(f"   output files are located in: {OUTPUT_DATA_DIR}")
    print(f"   - train_seq_len{MAX_SEQ_LEN}_sim_augmented.parquet")
    print(f"   - valid_seq_len{MAX_SEQ_LEN}_with_history.parquet")
    print(f"   - test_seq_len{MAX_SEQ_LEN}_with_history.parquet")
