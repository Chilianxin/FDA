from __future__ import annotations

from typing import List, Tuple
import numpy as np


def purged_embargo_splits(dates: List[str], train_months: int = 12, val_months: int = 3, test_months: int = 3, embargo_days: int = 20) -> List[Tuple[List[str], List[str], List[str]]]:
    # naive monthly buckets by YYYYMM
    months = sorted({d[:6] for d in dates})
    i = 0
    result = []
    while i + train_months + val_months + test_months <= len(months):
        train_m = months[i:i+train_months]
        val_m = months[i+train_months:i+train_months+val_months]
        test_m = months[i+train_months+val_months:i+train_months+val_months+test_months]
        # expand to dates
        train = [d for d in dates if d[:6] in train_m]
        val = [d for d in dates if d[:6] in val_m]
        test = [d for d in dates if d[:6] in test_m]
        # embargo naive removal: drop last embargo_days from train
        if len(train) > embargo_days:
            train = train[:-embargo_days]
        result.append((train, val, test))
        i += test_months
    return result

