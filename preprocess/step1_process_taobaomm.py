# This script implements a standard offline data processing pipeline for
# recommendation, specialized for building a "micro" (down-scaled) dataset.
# It consists of four core steps:
# Step 1: Sample alignment and filtering (Process Samples & Intersect)
# Purpose: process train_samples and test_samples.
# Label filtering: select rows via the check_is_click function (the current code
#   keeps the clicked rows).
# User sampling: if NUM_USERS is set, keep only the samples of the first N users.
# Item intersection: this is a critical step. It computes the intersection of the
#   item IDs appearing in the training and the test set.
# Output: the filtered train/test_samples.parquet and filtered_item_ids.txt.

# Step 2: User feature alignment (Process User Features)
# Purpose: process user_features.
# Using the "valid user list" determined in Step 1, extract only those users'
#   rows from the very large user feature table.
# Extract history sequences: parse the 150_2_180 column (assumed to hold the
#   historical behavior sequence) and collect every item ID occurring in any
#   sequence.
# Output: train/test_user_features.parquet.

# Step 3: Item pool construction and feature filtering (Process Item Features)
# Purpose: process item_features.
# Full item pool = target items (from the samples) + history items (from the
#   sequences).
# Rationale: the model must not only score the target item, it must also look up
#   embeddings for the items in the user's history sequence. If an item from a
#   history sequence is absent from the embedding table, the model raises an error.
# Filter the raw item feature table against this "full item pool".
# Output: item_features.parquet and final_item_ids.txt.

# Step 4: Wide table construction (Build Wide Table)
# Purpose: join samples, user features and item features into one wide table.
# DuckDB: run the SQL JOIN with the DuckDB engine. Compared with a pandas merge,
#   DuckDB manages memory much better when joining large files and is very fast.
# Sharding: split the result into several small Parquet files of SHARD_SIZE
#   (50000 rows) each. This works much better for parallel reads by a
#   PyTorch/TensorFlow DataLoader.

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import os
import numpy as np
import gc

SOURCE_DIR = "../../GLASS/data/TAOBAO_MM/raw"
MICRO_DIR = "./TAOBAO_MM/micro"

# User filtering control
#   - integer (e.g., 100000): keep only the first N users
#   - None                  : no filtering, keep all users
NUM_USERS = None  

# Item filtering control
#   - integer: cap the number of items in the intersection
#   - None   : no cap
NUM_ITEMS_SAMPLE = 10000
# Minimum item interaction frequency filter (applied to the training split only)
MIN_ITEM_FREQ = 5

SHARD_SIZE = 50000 

# [core setting] read the original samples files here
FILES = {
    "train_sample": "train_samples.parquet",  # original name: train_samples.parquet
    "test_sample":  "test_samples.parquet",   # original name: test_samples.parquet
    "train_feat":   "train_user_features.parquet",
    "test_feat":    "test_user_features.parquet",
    "item_feat":    ["item_features.parquet", "scl_embedding_int8_p90.parquet"]
}

def check_is_click(val):
    """Check whether the row is a click (label_0 == 0)."""
    if isinstance(val, (np.ndarray, list)):
        return val[0] == 0
    # Stay compatible with raw data where the field may not be a list.
    if isinstance(val, (int, float, np.number)):
        return val == 0
    return False

def save_parquet(df, path):
    if len(df) > 0:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        print(f"    [done] saved: {path} (Rows: {len(df)})")
    else:
        print(f"    [warn] data is empty, did not save {path}")


def step1_process_samples_and_intersect():
    print("\n[Step 1] Reading raw samples -> filtering users/clicks -> computing the item intersection ...")
    os.makedirs(MICRO_DIR, exist_ok=True)
    
    data_cache = {} 
    valid_users = {"train": set(), "test": set()}
    
    # --- 1.1 read & preprocess ---
    for mode, input_file in [("train", FILES["train_sample"]), ("test", FILES["test_sample"])]:
        src_path = os.path.join(SOURCE_DIR, input_file)
        if not os.path.exists(src_path):
            print(f"    [error] raw file not found: {src_path}")
            return None, None

        print(f"    Reading raw file: {input_file} ...")
        df = pd.read_parquet(src_path)
        
        # A. filter clicks
        if "label_0" in df.columns:
            df = df[df["label_0"].apply(check_is_click)]
        
        # B. filter users (Top N)
        if "129_1" in df.columns:
            unique_users = df["129_1"].unique()
            if NUM_USERS is not None and isinstance(NUM_USERS, int) and NUM_USERS > 0:
                selected_users = unique_users[:NUM_USERS] if len(unique_users) > NUM_USERS else unique_users
            else:
                selected_users = unique_users
            
            df = df[df["129_1"].isin(selected_users)]
            valid_users[mode] = set(selected_users) 
        
        data_cache[mode] = df
        print(f"    [done] {mode} cached after preprocessing: {len(df)} rows")
    # ========================================================
    # Filter out items with fewer than 5 interactions in the training split here
    # ========================================================
    if MIN_ITEM_FREQ is not None and MIN_ITEM_FREQ > 1:
        print(f"    Applying the low-frequency item filter (Min Freq >= {MIN_ITEM_FREQ}) ...")
        train_df = data_cache["train"]
        
        # 1. Count the occurrences of every item in the training split.
        item_counts = train_df["205"].value_counts()
        
        # 2. Find the item IDs that satisfy the threshold.
        high_freq_items = item_counts[item_counts >= MIN_ITEM_FREQ].index
        
        # 3. Report how much was filtered out.
        print(f"       original item count: {len(item_counts)} -> high-frequency item count: {len(high_freq_items)}")
        
        # 4. Filter the training DataFrame.
        # Only rows whose item is in high_freq_items are kept.
        data_cache["train"] = train_df[train_df["205"].isin(high_freq_items)]
        
        print(f"       training samples after filtering: {len(data_cache['train'])}")
    # --- 1.2 compute the item intersection ---
    print("    Computing the Train/Test item intersection ...")
    train_items = set(data_cache["train"]["205"].unique())
    test_items = set(data_cache["test"]["205"].unique())
    
    common_items = train_items.intersection(test_items)
    print(f"    number of items in the intersection: {len(common_items)}")
    
    if len(common_items) == 0:
        print("    [error] critical: Train and Test share no items!")
        return None, None

    # (optional) sampling
    if NUM_ITEMS_SAMPLE is not None and len(common_items) > NUM_ITEMS_SAMPLE:
        print(f"    [info] config active: randomly sampling {NUM_ITEMS_SAMPLE} items ...")
        np.random.seed(42)
        common_items = set(np.random.choice(list(common_items), size=NUM_ITEMS_SAMPLE, replace=False))

    # ==========================================
    # Save filtered_item_ids.txt
    # ==========================================
    filtered_txt_path = os.path.join(MICRO_DIR, "filtered_item_ids.txt")
    print(f"    Saving the target item intersection list to: {filtered_txt_path}")
    try:
        sorted_items = sorted(list(common_items))
        with open(filtered_txt_path, 'w', encoding='utf-8') as f:
            for item_id in sorted_items:
                f.write(f"{item_id}\n")
    except Exception as e:
        print(f"    [warn] failed to save filtered_item_ids.txt: {e}")

    # --- 1.3 filter the DataFrames and save ---
    target_item_ids = common_items
    
    for mode in ["train", "test"]:
        df = data_cache[mode]
        df_final = df[df["205"].isin(common_items)]
        
        save_parquet(df_final, os.path.join(MICRO_DIR, f"{mode}_samples.parquet"))
        
        valid_users[mode] = set(df_final["129_1"].unique())
        del df, df_final
        gc.collect()

    return valid_users, target_item_ids

def step2_process_user_features(valid_users):
    print("\n[Step 2] Processing user features & extracting the items of the history sequences ...")
    history_item_ids = set()
    
    for mode, input_file in [("train", FILES["train_feat"]), ("test", FILES["test_feat"])]:
        src_path = os.path.join(SOURCE_DIR, input_file)
        out_path = os.path.join(MICRO_DIR, f"{mode}_user_features.parquet")
        
        if not os.path.exists(src_path): 
            print(f"    [warn] skipping {input_file} (file does not exist)")
            continue
            
        print(f"    Reading features: {input_file} ...")
        df = pd.read_parquet(src_path)
        
        if "129_1" in df.columns and mode in valid_users:
            original_len = len(df)
            df = df[df["129_1"].isin(valid_users[mode])]
            print(f"       user alignment: {original_len} -> {len(df)}")
            
            if "150_2_180" in df.columns:
                seq_items = df["150_2_180"].explode().unique()
                seq_items = [x for x in seq_items if pd.notna(x)]
                history_item_ids.update(seq_items)
            
            save_parquet(df, out_path)
        
        del df
        gc.collect()
        
    print(f"    number of items contained in the history sequences: {len(history_item_ids)}")
    return history_item_ids

def step3_process_item_features(target_ids, history_ids):
    print("\n[Step 3] Building the item embedding table ...")
    final_item_ids = target_ids.union(history_ids)
    print(f"    final item pool size: {len(final_item_ids)} (Target: {len(target_ids)} + History: {len(history_ids)})")
    
    # Save the ID list.
    with open(os.path.join(MICRO_DIR, "final_item_ids.txt"), "w") as f:
        for iid in final_item_ids: f.write(f"{iid}\n")

    for filename in FILES["item_feat"]:
        src_path = os.path.join(SOURCE_DIR, filename)
        out_path = os.path.join(MICRO_DIR, filename)
        
        if os.path.exists(src_path):
            print(f"    Filtering file: {filename} ...")
            df = pd.read_parquet(src_path)
            if "205" in df.columns:
                df = df[df["205"].isin(final_item_ids)]
                save_parquet(df, out_path)
            del df
            gc.collect()

def step4_build_wide_table():
    print("\n[Step 4] Building the wide table with DuckDB (Join) ...")
    con = duckdb.connect()
    
    for mode in ["train", "test"]:
        sample_file = os.path.join(MICRO_DIR, f"{mode}_samples.parquet")
        user_file = os.path.join(MICRO_DIR, f"{mode}_user_features.parquet")
        item_file = os.path.join(MICRO_DIR, "item_features.parquet")
        
        if not (os.path.exists(sample_file) and os.path.exists(user_file)):
            print(f"    [warn] skipping {mode}: intermediate files are incomplete")
            continue

        output_dir = os.path.join(MICRO_DIR, mode)
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"    Processing the {mode} split ...")
        
        # Make sure item_features exists: a left join against a missing file would
        # be a problem, although DuckDB does allow a left join even on an empty
        # table. To stay robust, when item_file is absent we join the user table only.
        item_join_clause = f"LEFT JOIN '{item_file}' AS i ON s.\"205\" = i.\"205\"" if os.path.exists(item_file) else ""
        item_select_clause = ", i.\"206\", i.\"213\", i.\"214\"" if os.path.exists(item_file) else ""

        query = f"""
        SELECT 
            s."label_0", s."129_1",
            u."130_1", u."130_2", u."130_3", u."130_4", u."130_5",
            u."150_2_180", u."151_2_180", 
            s."205"
            {item_select_clause}
        FROM '{sample_file}' AS s
        LEFT JOIN '{user_file}' AS u ON s."129_1" = u."129_1"
        {item_join_clause}
        """
        
        try:
            reader = con.query(query).fetch_arrow_reader(batch_size=SHARD_SIZE)
            shard_id = 0
            while True:
                try:
                    batch = reader.read_next_batch()
                    if len(batch) == 0: break
                    table = pa.Table.from_batches([batch])
                    pq.write_table(table, os.path.join(output_dir, f"{mode}-shard-{shard_id:06d}.parquet"), compression='ZSTD')
                    shard_id += 1
                except StopIteration: break
            print(f"    [done] {mode} finished: {shard_id} shards")
        except Exception as e:
            print(f"    [error] DuckDB execution failed: {e}")

if __name__ == "__main__":
    valid_users_dict, target_items = step1_process_samples_and_intersect()
    if valid_users_dict:
        history_items = step2_process_user_features(valid_users_dict)
        step3_process_item_features(target_items, history_items)
        step4_build_wide_table()
        print("\n[done] all tasks finished")