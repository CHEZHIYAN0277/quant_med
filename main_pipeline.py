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
# Select candidates
# ------------------------

scores = model.predict_proba(X)[:,1]

top_idx = np.argsort(scores)[-120:]

X_top = X[top_idx]
y_top = y[top_idx]


# ------------------------
# PCA for quantum stage
# ------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_top)

pca = PCA(n_components=3)

X_quantum = pca.fit_transform(X_scaled)

X_quantum = X_quantum[:40]
y_quantum = y_top[:40]


# ------------------------
# Quantum refinement
# ------------------------

print("Running quantum stage...")

q_scores = quantum_refinement(X_quantum, y_quantum)

print("Quantum stage completed")

print("Top predicted molecules:")
print(df.iloc[top_idx[:10]])