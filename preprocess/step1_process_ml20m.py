"""
process_ml20m.py
========================================================
ML-20M data preprocessing script

- Read ratings.csv / movies.csv / tags.csv
- Build user-item interaction sequences (sorted by ascending timestamp)
- Map movieId -> contiguous internal ID
- Merge item metadata (title, genres, tag)
- Leave-one-out split: last item -> test, second to last -> validation,
  the rest -> training
- Truncate sequences to a maximum length (default 50)
- Write intermediate files for the downstream pipeline
"""
import os
import sys
import argparse
import json
import random
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= Top-level configuration =================
# Relative defaults; override on the command line or via DRIFT_DATA_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")

DEFAULT_DATA_DIR = os.path.join(_DATA_ROOT, "raw", "ml-20m")
DEFAULT_OUTPUT_DIR = os.path.join(_DATA_ROOT, "ml20m_processed")
DEFAULT_MAX_LEN = 50
MIN_INTERACTIONS = 5  # leave-one-out needs at least 5 interactions (3 train + 1 valid + 1 test)
SEED = 2024


def parse_args():
    parser = argparse.ArgumentParser(description="ML-20M Data Preprocessing")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="ML-20M raw data directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for processed data")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN,
                        help="Maximum sequence length per user (default: 50)")
    parser.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS,
                        help="Minimum interactions per user for leave-one-out (default: 5)")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "process_ml20m.log")),
        ],
        force=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)
    logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info(f"[Config] data_dir       : {args.data_dir}")
    logger.info(f"[Config] output_dir     : {args.output_dir}")
    logger.info(f"[Config] max_len        : {args.max_len}")
    logger.info(f"[Config] min_interactions: {args.min_interactions}")
    logger.info("=" * 60)

    # ---- Step 1: read the raw data ----
    logger.info("Step 1: Reading raw data...")

    ratings_path = os.path.join(args.data_dir, "ratings.csv")
    movies_path = os.path.join(args.data_dir, "movies.csv")
    tags_path = os.path.join(args.data_dir, "tags.csv")

    logger.info(f"  Reading ratings from {ratings_path}...")
    df_ratings = pd.read_csv(ratings_path)
    logger.info(f"  Ratings shape: {df_ratings.shape}")

    logger.info(f"  Reading movies from {movies_path}...")
    df_movies = pd.read_csv(movies_path)
    logger.info(f"  Movies shape: {df_movies.shape}")

    logger.info(f"  Reading tags from {tags_path}...")
    df_tags = pd.read_csv(tags_path)
    logger.info(f"  Tags shape: {df_tags.shape}")

    # ---- Step 2: filter ratings, keeping only records with a known movieId ----
    logger.info("Step 2: Filtering ratings...")
    # Collect every movieId that appears in movies.csv.
    valid_movie_ids = set(df_movies["movieId"].unique())
    df_ratings = df_ratings[df_ratings["movieId"].isin(valid_movie_ids)].copy()
    logger.info(f"  After filtering by valid movies: {df_ratings.shape}")

    # ---- Step 3: map movieId -> contiguous internal ID ----
    logger.info("Step 3: Building movieId -> internal ID mapping...")
    # Only map movieIds that actually occur in ratings.
    rated_movie_ids = sorted(df_ratings["movieId"].unique())
    movie_id_map = {mid: idx for idx, mid in enumerate(rated_movie_ids)}
    id_to_movie_map = {idx: mid for mid, idx in movie_id_map.items()}

    logger.info(f"  Total unique rated movies: {len(movie_id_map)}")

    # Persist the mappings.
    with open(os.path.join(args.output_dir, "movie_id_map.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in movie_id_map.items()}, f)
    with open(os.path.join(args.output_dir, "id_to_movie_map.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in id_to_movie_map.items()}, f)
    logger.info("  movie_id_map.json and id_to_movie_map.json saved.")

    # ---- Step 4: merge the item metadata ----
    logger.info("Step 4: Merging item metadata...")

    # Handle tags: deduplicate and concatenate all tags of the same movieId.
    df_tags_clean = df_tags.dropna(subset=["tag"]).copy()
    df_tags_clean["tag"] = df_tags_clean["tag"].astype(str).str.strip()
    df_tags_clean = df_tags_clean[df_tags_clean["tag"] != ""]
    df_tags_grouped = df_tags_clean.groupby("movieId")["tag"].apply(
        lambda x: " | ".join(sorted(set(x)))
    ).reset_index()
    df_tags_grouped.columns = ["movieId", "tags"]

    # Merge movies + tags.
    df_item_meta = df_movies.merge(df_tags_grouped, on="movieId", how="left")

    # Fill in the missing values.
    df_item_meta["title"] = df_item_meta["title"].fillna("Unknown")
    df_item_meta["genres"] = df_item_meta["genres"].fillna("Unknown")
    df_item_meta["tags"] = df_item_meta["tags"].fillna("")

    # Keep only the items that occur in ratings.
    df_item_meta = df_item_meta[df_item_meta["movieId"].isin(rated_movie_ids)].copy()

    # Attach the internal ID.
    df_item_meta["item_id"] = df_item_meta["movieId"].map(movie_id_map)

    # Build the unified text field (used to generate text embeddings).
    def build_item_text(row):
        parts = []
        if row["title"] and row["title"] != "Unknown":
            parts.append(f"Title: {row['title']}")
        if row["genres"] and row["genres"] != "Unknown":
            parts.append(f"Genres: {row['genres']}")
        if row["tags"]:
            parts.append(f"Tags: {row['tags']}")
        return " ".join(parts) if parts else "Unknown item"

    df_item_meta["item_text"] = df_item_meta.apply(build_item_text, axis=1)

    # Sort by item_id.
    df_item_meta = df_item_meta.sort_values("item_id").reset_index(drop=True)

    # Save the item metadata.
    df_item_meta.to_parquet(os.path.join(args.output_dir, "item_metadata.parquet"), index=False)
    logger.info(f"  Item metadata saved: {len(df_item_meta)} items")

    # ---- Step 5: build the user interaction sequences ----
    logger.info("Step 5: Building user interaction sequences...")

    # Map movieId -> item_id.
    df_ratings["item_id"] = df_ratings["movieId"].map(movie_id_map)

    # Sort by userId, then timestamp.
    df_ratings = df_ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    # Group by userId to build the sequences.
    user_sequences = {}
    for uid, group in tqdm(df_ratings.groupby("userId"), desc="Building sequences"):
        seq = group["item_id"].tolist()
        user_sequences[int(uid)] = seq

    logger.info(f"  Total users before filtering: {len(user_sequences)}")

    # ---- Step 6: drop users with too few interactions ----
    logger.info("Step 6: Filtering users with insufficient interactions...")
    filtered_users = {
        uid: seq for uid, seq in user_sequences.items()
        if len(seq) >= args.min_interactions
    }
    skipped = len(user_sequences) - len(filtered_users)
    logger.info(f"  Skipped {skipped} users with < {args.min_interactions} interactions")
    logger.info(f"  Remaining users: {len(filtered_users)}")
    user_sequences = filtered_users

    # ---- Step 7: truncate the sequences (max_len) ----
    logger.info(f"Step 7: Truncating sequences to max_len={args.max_len}...")
    for uid in user_sequences:
        if len(user_sequences[uid]) > args.max_len:
            user_sequences[uid] = user_sequences[uid][-args.max_len:]

    # ---- Step 8: leave-one-out split ----
    logger.info("Step 8: Leave-one-out split...")

    train_records = []
    valid_records = []
    test_records = []

    for uid, seq in tqdm(user_sequences.items(), desc="Splitting"):
        # Test set: the last item.
        test_target = seq[-1]
        # Validation set: the second to last item.
        valid_target = seq[-2]
        # Training history: everything except the last two items.
        train_history = seq[:-2]

        # Training samples: every position of the history acts as a target.
        for i in range(1, len(train_history)):
            hist = train_history[:i]
            target = train_history[i]
            train_records.append({
                "user_id": uid,
                "history": hist[-args.max_len:],
                "target": target,
            })

        # Validation sample.
        valid_records.append({
            "user_id": uid,
            "history": train_history[-args.max_len:],
            "target": valid_target,
        })

        # Test sample: the validation target is appended to the history, because
        # at test time that interaction has already been observed by the model.
        test_records.append({
            "user_id": uid,
            "history": (train_history + [valid_target])[-args.max_len:],
            "target": test_target,
        })

    logger.info(f"  Train samples: {len(train_records)}")
    logger.info(f"  Valid samples: {len(valid_records)}")
    logger.info(f"  Test samples: {len(test_records)}")

    # ---- Step 9: save the split data ----
    logger.info("Step 9: Saving split data...")

    # Save a more compact format as well: one row per user holding the full
    # sequence, so that downstream stages can re-derive their own split.
    user_data = []
    for uid, seq in user_sequences.items():
        user_data.append({
            "user_id": uid,
            "sequence": seq,
        })
    df_users = pd.DataFrame(user_data)
    df_users.to_parquet(os.path.join(args.output_dir, "user_sequences.parquet"), index=False)
    logger.info(f"  User sequences saved: {len(df_users)} users")

    # Save the train/valid/test splits (as parquet as well).
    df_train = pd.DataFrame(train_records)
    df_valid = pd.DataFrame(valid_records)
    df_test = pd.DataFrame(test_records)

    df_train.to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    df_valid.to_parquet(os.path.join(args.output_dir, "valid.parquet"), index=False)
    df_test.to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)

    logger.info(f"  train.parquet saved: {len(df_train)} rows")
    logger.info(f"  valid.parquet saved: {len(df_valid)} rows")
    logger.info(f"  test.parquet saved: {len(df_test)} rows")

    # ---- Step 10: save the user ID mapping ----
    user_ids = sorted(user_sequences.keys())
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    with open(os.path.join(args.output_dir, "user_id_map.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in user_id_map.items()}, f)
    logger.info("  user_id_map.json saved.")

    # ---- Statistics ----
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    logger.info("=" * 60)
    logger.info("Data Statistics:")
    logger.info(f"  Total users: {len(user_sequences)}")
    logger.info(f"  Total items: {len(movie_id_map)}")
    logger.info(f"  Avg sequence length: {np.mean(seq_lengths):.2f}")
    logger.info(f"  Median sequence length: {np.median(seq_lengths):.2f}")
    logger.info(f"  Max sequence length: {max(seq_lengths)}")
    logger.info(f"  Min sequence length: {min(seq_lengths)}")
    logger.info("=" * 60)
    logger.info("[done] process_ml20m.py done!")


if __name__ == "__main__":
    main()
