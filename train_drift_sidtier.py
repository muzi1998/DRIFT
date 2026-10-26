#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main entry (variant A): encoder prefix = [SID-tier ; recent window].

Preference-shift subsequence learning for long-sequence generative recommendation.
The long-term history is adaptively partitioned into ``K`` preference
sub-sequences; their representations ``h_uk`` are injected into every encoder block
via cross-attention and into the decoder as the gated second cross-attention path.

Encoder prefix::

    h_0 = Dropout([ e_sidtier(1) ; e_recent(L_rec) ])

Here the SID-tier token summarizes the coarse semantic distribution of the user
history, while the sub-sequence structure enters through the in-block
cross-attention rather than the prefix, which keeps the prefix compact and avoids
making the two signals redundant.

See ``train_core.py`` for the full method description, the shared training loop and
the loss definitions; ``train_drift_recent.py`` is the counterpart variant whose
prefix contains the recent window only.

Example (KuaiRec)::

    python train_drift_sidtier.py \
        --data_root kuairec_processed --codebook 256,256,256 --ts_suffix \
        --max_len 1000 --recent_window_r 20 --K 20\
        --batch_size 512 --num_epochs 100 --out_dir outputs/kuairec_sidtier
"""

import argparse

import paths
import train_core

VARIANT = 'drift_sidtier'


def build_parser():
    ap = argparse.ArgumentParser(
        description="Preference-shift subsequence learning; "
                    "encoder prefix = [SID-tier ; recent window]")
    train_core.add_common_arguments(ap)
    ap.add_argument('--out_dir', default=paths.output_path(VARIANT),
                    help='directory for the best/summary JSON files')
    return ap


def main():
    args = build_parser().parse_args()
    train_core.run_training(
        args,
        variant=VARIANT,
        use_sidtier_prefix=True,      # prefix = [sidtier ; recent]
        log_tag='drift-sidtier')


if __name__ == "__main__":
    main()
