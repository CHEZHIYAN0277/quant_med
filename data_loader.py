import pandas as pd
from chembl_webresource_client.new_client import new_client

def fetch_dataset(limit=1000):

    activity = new_client.activity

    records = activity.filter(
        target_chembl_id="CHEMBL284",
        standard_type="IC50"
    ).only(["canonical_smiles","standard_value"])

    data = []

    for r in records:

        if len(data) >= limit:
            break

        if r["canonical_smiles"] and r["standard_value"]:

            try:
                ic50 = float(r["standard_value"])
            except:
                continue

            label = 1 if ic50 <= 1000 else 0

            data.append({
                "SMILES": r["canonical_smiles"],
                "IC50": ic50,
                "Activity": label
            })

    df = pd.DataFrame(data).drop_duplicates("SMILES")

    return df