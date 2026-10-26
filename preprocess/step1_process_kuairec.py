# -*- coding: utf-8 -*-
"""
process_kuairec.py
==========================================================
KuaiRec dataset preprocessing script
----------------------------------------------------------
Overview:
    1. Read the raw KuaiRec files from the source directory:
         - big_matrix.csv               : user-video interactions
         - item_categories.csv          : video category information
         - item_daily_features.csv      : per-day video features
         - kuairec_caption_category.csv : video text / tag / category information
    2. Join the interactions with the item-side features on video_id to form a
       single wide table.
    3. Split per user with Leave-One-Out:
         - most recent interaction        -> test set
         - second most recent interaction -> validation set
         - all remaining interactions     -> training set
    4. Save the merged table and the three splits to disk (CSV + Parquet).
==========================================================
"""

import os
import sys
import ast
import gc
import time
import warnings

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ============================================================
# Paths and constants
# ============================================================
# Directory holding the raw KuaiRec release. Override with KUAIREC_RAW_DIR, or
# place the raw CSV files under <repo>/data/raw/KuaiRec/data/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")

SOURCE_DIR = os.environ.get("KUAIREC_RAW_DIR") or os.path.join(
    _DATA_ROOT, "raw", "KuaiRec", "data")

# Output directory consumed by step4_build_kuairec.py.
OUTPUT_DIR = os.environ.get("KUAIREC_OUT_DIR") or os.path.join(
    _DATA_ROOT, "kuairec_processed")

# Raw file names
FILE_INTERACTION = "big_matrix.csv"
FILE_ITEM_CATEG = "item_categories.csv"
FILE_ITEM_DAILY = "item_daily_features.csv"
FILE_ITEM_CAPTION = "kuairec_caption_category.csv"

# Columns kept from the interaction table
INTERACTION_COLS = ["user_id", "video_id", "timestamp"]

# Item-side columns to keep (collected across the three item tables)
ITEM_FEATURE_COLS = [
    "author_id",
    "video_type",
    "upload_type",
    "video_duration",
    "video_width",
    "video_height",
    "music_id",
    "video_tag_name",
    "caption",
    "topic_tag",
    "first_level_category_name",
    "second_level_category_name",
    "feat",
]


# ============================================================
# Utility functions
# ============================================================
def _log(msg: str) -> None:
    """Unified logging helper with a timestamp prefix."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _safe_read_csv(path: str, usecols=None, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file defensively:
      - any column listed in usecols that is absent from the file is skipped
        automatically instead of raising.
    """
    if not os.path.exists(path):
        
        raise FileNotFoundError(f"File not found: {path}")

    # Read only the header first to learn the real column names, so that a
    # non-existent entry in usecols does not raise an error.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if usecols is not None:
        usecols = [c for c in usecols if c in header]
        if len(usecols) == 0:
            # If none of the requested columns exist, fall back to reading all.
            usecols = None

    # Try the default C engine first; if it hits a buffer overflow / tokenize
    # error, fall back to the python engine with a more permissive parsing
    # policy so that files containing very long text or unusual quoting still
    # load successfully.
    try:
        df = pd.read_csv(path, usecols=usecols, **kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        _log(f"    [warn] default engine failed to parse ({type(e).__name__}): {e}; falling back to the python engine ...")
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("low_memory", None)
        fallback_kwargs["engine"] = "python"
        fallback_kwargs["on_bad_lines"] = "skip"
        if "encoding" not in fallback_kwargs:
            fallback_kwargs["encoding"] = "utf-8"
        df = pd.read_csv(path, usecols=usecols, **fallback_kwargs)
    return df


def _dedup_on_video(df: pd.DataFrame, key: str = "video_id") -> pd.DataFrame:
    """Deduplicate an item feature table on video_id, keeping the first row."""
    if key not in df.columns:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    _log(f"    dedup {key}: {before} -> {len(df)}")
    return df


# ============================================================
# Step 1: load the interaction data
# ============================================================
def load_interactions() -> pd.DataFrame:
    """Read big_matrix.csv and extract (user_id, video_id, timestamp)."""
    _log("[Step 1] Reading interaction data from big_matrix.csv ...")
    path = os.path.join(SOURCE_DIR, FILE_INTERACTION)
    df = _safe_read_csv(path, usecols=INTERACTION_COLS)

    # Basic cleaning: drop rows missing any of the required fields.
    df = df.dropna(subset=["user_id", "video_id", "timestamp"]).reset_index(drop=True)

    # Normalize dtypes.
    df["user_id"] = df["user_id"].astype(np.int64)
    df["video_id"] = df["video_id"].astype(np.int64)
    # In KuaiRec the timestamp is usually a float second-level epoch value; cast
    # it to float64 so that sorting behaves consistently.
    df["timestamp"] = df["timestamp"].astype(np.float64)

    _log(f"    interactions: {len(df):,} | users: {df['user_id'].nunique():,} | videos: {df['video_id'].nunique():,}")
    return df


# ============================================================
# Step 2: load and merge the item features
# ============================================================
def load_item_features() -> pd.DataFrame:
    """
    Pull the requested fields from the three item tables and merge them on
    video_id:
        - item_daily_features.csv : author_id / video_type / upload_type /
                                    video_duration / video_width / video_height / music_id
        - item_categories.csv     : feat (category list)
        - kuairec_caption_category.csv :
                                    video_tag_name / caption / topic_tag /
                                    first_level_category_name / second_level_category_name
    Returns a wide item table keyed by video_id (exactly one row per video_id).
    """
    _log("[Step 2] Reading item-side features (three tables) ...")

    # ---- 2.1 item_daily_features.csv ----
    daily_cols = [
        "video_id",
        "author_id",
        "video_type",
        "upload_type",
        "video_duration",
        "video_width",
        "video_height",
        "music_id",
    ]
    daily_path = os.path.join(SOURCE_DIR, FILE_ITEM_DAILY)
    _log(f"    Reading {FILE_ITEM_DAILY} ...")
    df_daily = _safe_read_csv(daily_path, usecols=daily_cols)
    # The daily table contains one row per video per day; keep the earliest row
    # of each video as its static attribute set.
    df_daily = _dedup_on_video(df_daily, key="video_id")

    # ---- 2.2 item_categories.csv ----
    categ_cols = ["video_id", "feat"]
    categ_path = os.path.join(SOURCE_DIR, FILE_ITEM_CATEG)
    _log(f"    Reading {FILE_ITEM_CATEG} ...")
    df_categ = _safe_read_csv(categ_path, usecols=categ_cols)
    df_categ = _dedup_on_video(df_categ, key="video_id")

    # ---- 2.3 kuairec_caption_category.csv ----
    caption_cols = [
        "video_id",
        "video_tag_name",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
    ]
    cap_path = os.path.join(SOURCE_DIR, FILE_ITEM_CAPTION)
    _log(f"    Reading {FILE_ITEM_CAPTION} ...")
    df_cap = _safe_read_csv(cap_path, usecols=caption_cols)
    df_cap = _dedup_on_video(df_cap, key="video_id")

    # ---- 2.4 unify the video_id dtype before merging ----
    # When a CSV falls back to the python engine, video_id may be parsed as an
    # object column. To avoid a dtype mismatch during merge, coerce it to int64
    # here; rows that cannot be parsed as integers are dropped.
    for _name, _df in [("daily", df_daily), ("categ", df_categ), ("cap", df_cap)]:
        before = len(_df)
        _df["video_id"] = pd.to_numeric(_df["video_id"], errors="coerce")
        _df.dropna(subset=["video_id"], inplace=True)
        _df["video_id"] = _df["video_id"].astype(np.int64)
        if len(_df) != before:
            _log(f"    [info] normalized video_id in the {_name} table: {before} -> {len(_df)}")
    # Deduplicate once more in case the dtype conversion introduced duplicates.
    df_cap = df_cap.drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)

    # ---- 2.5 merge the three tables ----
    _log("    Merging the three item feature tables (on=video_id, how=outer) ...")
    df_item = df_daily.merge(df_categ, on="video_id", how="outer")
    df_item = df_item.merge(df_cap, on="video_id", how="outer")

    # Keep only the requested fields plus the primary key.
    keep_cols = ["video_id"] + [c for c in ITEM_FEATURE_COLS if c in df_item.columns]
    missing = [c for c in ITEM_FEATURE_COLS if c not in df_item.columns]
    if missing:
        _log(f"    [warn] fields missing from the source data (skipped): {missing}")
    df_item = df_item[keep_cols].reset_index(drop=True)

    _log(f"    [done] item feature table size: {df_item.shape[0]:,} rows x {df_item.shape[1]} columns")
    return df_item


# ============================================================
# Step 3: merge interactions with item features
# ============================================================
def merge_interaction_with_items(df_inter: pd.DataFrame,
                                 df_item: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the item features onto the interactions on video_id to build the
    full sample table. The left join guarantees that every interaction row is
    kept; videos without features get NaN in the corresponding columns.
    """
    _log("[Step 3] Merging interactions with item features (on=video_id, how=left) ...")
    df = df_inter.merge(df_item, on="video_id", how="left")
    _log(f"    [done] rows after merge: {len(df):,} | columns: {df.shape[1]}")

    # Report the missing-value rate of key columns to ease data quality checks.
    for c in ["author_id", "first_level_category_name", "caption"]:
        if c in df.columns:
            miss_rate = df[c].isna().mean() * 100
            _log(f"       missing rate - {c}: {miss_rate:.2f}%")

    return df


# ============================================================
# Step 4: Leave-One-Out split
# ============================================================
def leave_one_out_split(df: pd.DataFrame):
    """
    Leave-One-Out split:
      - sort each user's interactions by ascending timestamp;
      - the last interaction         -> test set;
      - the second to last           -> validation set;
      - all remaining interactions   -> training set;
      - users with fewer than 3 interactions: everything goes to the training
        set (they cannot contribute a validation/test example).

    Returns (train_df, valid_df, test_df)
    """
    _log("[Step 4] Leave-One-Out split into train / valid / test ...")

    # Stable sort: first by user_id, then by ascending timestamp.
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Compute a "counting from the end" rank within each user group:
    #   rank_from_last == 1 -> most recent interaction (test)
    #   rank_from_last == 2 -> second most recent interaction (validation)
    #   otherwise           -> training
    # Using groupby.cumcount in reverse is more efficient than apply.
    g = df.groupby("user_id", sort=False)
    # Total number of interactions per user.
    inter_cnt = g["timestamp"].transform("size")
    # Forward position of each interaction within its user (0-based).
    pos = g.cumcount()
    # Distance from the user's most recent interaction (0 means most recent).
    rank_from_last = inter_cnt - 1 - pos

    # Only apply leave-one-out to users with >= 3 interactions; otherwise every
    # row becomes training data.
    eligible = inter_cnt >= 3

    test_mask = eligible & (rank_from_last == 0)
    valid_mask = eligible & (rank_from_last == 1)
    train_mask = ~(test_mask | valid_mask)  # everything else is training

    train_df = df[train_mask].reset_index(drop=True)
    valid_df = df[valid_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    n_users_total = df["user_id"].nunique()
    n_users_eligible = df.loc[eligible, "user_id"].nunique()
    _log(f"    users: {n_users_total:,} | users included in the LOO split: {n_users_eligible:,}")
    _log(f"    Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}")

    # Sanity check: each user must appear exactly once in valid/test.
    assert valid_df["user_id"].is_unique, "The validation set contains multiple interactions from the same user!"
    assert test_df["user_id"].is_unique, "The test set contains multiple interactions from the same user!"

    return train_df, valid_df, test_df


# ============================================================
# Step 5: save the results
# ============================================================
def save_splits(merged: pd.DataFrame,
                train_df: pd.DataFrame,
                valid_df: pd.DataFrame,
                test_df: pd.DataFrame) -> None:
    """Save the merged table and train/valid/test splits as CSV and Parquet."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log(f"[Step 5] Saving results to {OUTPUT_DIR} ...")

    to_save = {
        "merged_all": merged,
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    for name, d in to_save.items():
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        pq_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
        d.to_csv(csv_path, index=False)
        try:
            d.to_parquet(pq_path, index=False)
        except Exception as e:
            # Some environments lack pyarrow/fastparquet; keep only the CSV then.
            _log(f"    [warn] failed to write parquet (keeping csv only): {e}")
        _log(f"    {name:<10s} -> {csv_path}  ({len(d):,} rows)")


# ============================================================
# Main pipeline
# ============================================================
def main():
    t0 = time.time()

    # Step 1: load interactions
    df_inter = load_interactions()

    # Step 2: load and merge item features
    df_item = load_item_features()

    # Step 3: merge interactions with item features
    df_merged = merge_interaction_with_items(df_inter, df_item)

    # Release the large intermediate objects that are no longer needed.
    del df_inter, df_item
    gc.collect()

    # Step 4: Leave-One-Out split
    train_df, valid_df, test_df = leave_one_out_split(df_merged)

    # Step 5: save
    save_splits(df_merged, train_df, valid_df, test_df)

    _log(f"[done] all finished in {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
