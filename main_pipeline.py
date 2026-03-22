import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import fetch_dataset
from classical_screening import train_classifier
from quantum_refinement import quantum_refinement

from rdkit import Chem
from rdkit.Chem import AllChem


print("Loading dataset...")

df = fetch_dataset()

print("Dataset size:", len(df))


# ------------------------
# Fingerprint generation
# ------------------------

def fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    return np.array(fp)


X = np.array([fingerprint(s) for s in df.SMILES])
y = df.Activity.values


# ------------------------
# Classical model
# ------------------------

model, acc = train_classifier(X, y)

print("Classical accuracy:", acc)


# ------------------------
# Select balanced candidates for quantum stage
# ------------------------

scores = model.predict_proba(X)[:,1]

# Get indices for each class from the full dataset
all_active = np.where(y == 1)[0]
all_inactive = np.where(y == 0)[0]

# Rank by classifier confidence within each class
active_ranked = all_active[np.argsort(scores[all_active])[::-1]]
inactive_ranked = all_inactive[np.argsort(scores[all_inactive])]

n_per_class = min(20, len(active_ranked), len(inactive_ranked))

selected_idx = np.concatenate([active_ranked[:n_per_class], inactive_ranked[:n_per_class]])
np.random.shuffle(selected_idx)

X_selected = X[selected_idx]
y_selected = y[selected_idx]

print(f"Quantum subset: {n_per_class} active + {n_per_class} inactive = {2*n_per_class} molecules")


# ------------------------
# PCA for quantum stage
# ------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_selected)

n_components = min(3, X_scaled.shape[0], X_scaled.shape[1])

pca = PCA(n_components=n_components)

X_quantum = pca.fit_transform(X_scaled)
y_quantum = y_selected


# ------------------------
# Quantum refinement
# ------------------------

print("Running quantum stage...")

q_scores = quantum_refinement(X_quantum, y_quantum)

print("Quantum stage completed")

print("Top predicted molecules:")
print(df.iloc[selected_idx[:10]])