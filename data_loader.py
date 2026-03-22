import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

CACHE_FILE = os.path.join(os.path.dirname(__file__), "chembl_dpp4_dataset.csv")


def fetch_dataset(limit=None, force_refresh=False):
    """
    Fetch DPP-4 (CHEMBL284) IC50 activity data from ChEMBL.

    On first run the full dataset is downloaded and cached locally
    as 'chembl_dpp4_dataset.csv'.  Subsequent calls load directly
    from the cached CSV (unless force_refresh=True).

    Parameters
    ----------
    limit : int or None
        Maximum number of records to return.  None = return everything.
    force_refresh : bool
        If True, re-download from ChEMBL even if the cache exists.
    """

    # ── Use cached file if available ──────────────────────────────
    if os.path.exists(CACHE_FILE) and not force_refresh:
        print(f"Loading cached dataset from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
        if limit is not None:
            df = df.head(limit)
        return df

    # ── Download from ChEMBL API ──────────────────────────────────
    print("Downloading DPP-4 (CHEMBL284) dataset from ChEMBL …")

    activity = new_client.activity

    records = activity.filter(
        target_chembl_id="CHEMBL284",
        standard_type="IC50"
    ).only(["canonical_smiles", "standard_value"])

    data = []

    for r in records:

        if r["canonical_smiles"] and r["standard_value"]:

            try:
                ic50 = float(r["standard_value"])
            except (ValueError, TypeError):
                continue

            label = 1 if ic50 <= 1000 else 0

            data.append({
                "SMILES": r["canonical_smiles"],
                "IC50": ic50,
                "Activity": label
            })

    df = pd.DataFrame(data).drop_duplicates("SMILES")

    # ── Cache to disk ─────────────────────────────────────────────
    df.to_csv(CACHE_FILE, index=False)
    print(f"Cached {len(df)} unique molecules → {CACHE_FILE}")

    if limit is not None:
        df = df.head(limit)

    return df