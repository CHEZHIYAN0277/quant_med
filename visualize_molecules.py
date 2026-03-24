"""
3D Quantum Molecular Visualizer — Data Generator
=================================================
Runs the pipeline, selects the top molecules by quantum score,
generates 3D coordinates with RDKit, and produces a self-contained
interactive HTML viewer (molecule_viewer.html).

Usage:
    python visualize_molecules.py
"""

import json
import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import fetch_dataset
from classical_screening import train_classifier
from quantum_refinement import quantum_refinement


# ── Element visual properties (for the HTML viewer) ──────────────────────

ELEMENT_COLORS = {
    "C":  "#00e5ff", "N":  "#4a6cff", "O":  "#ff3d5a",
    "S":  "#ffe14d", "F":  "#7cff4d", "Cl": "#4dff9e",
    "Br": "#ff8a4d", "P":  "#ff9dff", "I":  "#c44dff",
    "H":  "#aaaaaa",
}

# ── Known DPP-4 inhibitors for reference comparison ─────────────────

REFERENCE_MOLECULES = [
    {
        "name": "Sitagliptin",
        "smiles": "Fc1cc(c(F)cc1F)C[C@@H](N)CC(=O)N1CCn2c(nnc2C(F)(F)F)C1",
        "ic50": 18.0,
        "description": "First FDA-approved DPP-4 inhibitor (Januvia)",
    },
    {
        "name": "Vildagliptin",
        "smiles": "O=C(CN1CCC[C@@H]1C#N)[NH]C1CC2CC(C1)C(O)C2",
        "ic50": 3.5,
        "description": "Second-generation DPP-4 inhibitor (Galvus)",
    },
    {
        "name": "Saxagliptin",
        "smiles": "N#C[C@H]1CC(O)(CC1N)C12CC3CC(CC(O)(C3)C1)C2",
        "ic50": 1.3,
        "description": "Potent DPP-4 inhibitor (Onglyza)",
    },
]

ELEMENT_RADII = {
    "C": 0.40, "N": 0.38, "O": 0.36, "S": 0.50, "F": 0.32,
    "Cl": 0.48, "Br": 0.54, "P": 0.52, "I": 0.58, "H": 0.20,
}


# ── 3D coordinate generation ────────────────────────────────────────────

def mol_to_3d_data(smiles):
    """Convert SMILES → 3D atom positions + bond list."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            return None

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conf = mol.GetConformer()
    atoms = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atoms.append({
            "element": symbol,
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
            "z": round(pos.z, 3),
            "color": ELEMENT_COLORS.get(symbol, "#ffffff"),
            "radius": ELEMENT_RADII.get(symbol, 0.35),
        })

    bonds = []
    for bond in mol.GetBonds():
        order = int(bond.GetBondTypeAsDouble())
        bonds.append({
            "start": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(),
            "order": order,
        })

    return {"atoms": atoms, "bonds": bonds}


# ── MCS-based comparison data ────────────────────────────────────────────

def compute_comparison_data(molecules_data, reference_data):
    """Compute MCS-based structural comparisons between candidates and references.
    Returns a dict mapping 'cand_idx-ref_idx' to highlight data."""
    comparisons = {}

    for ci, cand in enumerate(molecules_data):
        cand_mol = Chem.MolFromSmiles(cand["smiles"])
        if cand_mol is None:
            continue

        for ri, ref in enumerate(reference_data):
            ref_mol = Chem.MolFromSmiles(ref["smiles"])
            if ref_mol is None:
                continue

            try:
                mcs_result = rdFMCS.FindMCS(
                    [cand_mol, ref_mol],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                    timeout=5,
                )
                mcs_smarts = mcs_result.smartsString
                if not mcs_smarts:
                    raise ValueError("Empty MCS")

                mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                if mcs_mol is None:
                    raise ValueError("Invalid MCS SMARTS")

                # Find atoms in candidate that match MCS
                cand_match = cand_mol.GetSubstructMatch(mcs_mol)
                ref_match = ref_mol.GetSubstructMatch(mcs_mol)

                cand_common = set(cand_match) if cand_match else set()
                ref_common = set(ref_match) if ref_match else set()

                # All heavy atom indices
                cand_all = {a.GetIdx() for a in cand_mol.GetAtoms() if a.GetSymbol() != 'H'}
                ref_all = {a.GetIdx() for a in ref_mol.GetAtoms() if a.GetSymbol() != 'H'}

                cand_unique = sorted(cand_all - cand_common)
                ref_unique = sorted(ref_all - ref_common)

                # Build text summary of differences
                cand_unique_elements = {}
                for idx in cand_unique:
                    el = cand_mol.GetAtomWithIdx(idx).GetSymbol()
                    cand_unique_elements[el] = cand_unique_elements.get(el, 0) + 1

                ref_unique_elements = {}
                for idx in ref_unique:
                    el = ref_mol.GetAtomWithIdx(idx).GetSymbol()
                    ref_unique_elements[el] = ref_unique_elements.get(el, 0) + 1

                adds = ", ".join(f"+{count} {el}" for el, count in sorted(cand_unique_elements.items()))
                removes = ", ".join(f"-{count} {el}" for el, count in sorted(ref_unique_elements.items()))
                summary_parts = []
                if adds:
                    summary_parts.append(f"Candidate adds: {adds}")
                if removes:
                    summary_parts.append(f"Reference has: {removes}")
                summary = " | ".join(summary_parts) if summary_parts else "Identical scaffolds"

                comparisons[f"{ci}-{ri}"] = {
                    "cand_unique": cand_unique,
                    "ref_unique": ref_unique,
                    "common_count": len(cand_common),
                    "cand_unique_count": len(cand_unique),
                    "ref_unique_count": len(ref_unique),
                    "mcs_atoms": mcs_result.numAtoms,
                    "summary": summary,
                }
            except Exception:
                comparisons[f"{ci}-{ri}"] = {
                    "cand_unique": [],
                    "ref_unique": [],
                    "common_count": 0,
                    "cand_unique_count": 0,
                    "ref_unique_count": 0,
                    "mcs_atoms": 0,
                    "summary": "Could not compute MCS",
                }

    return comparisons


# ── Run pipeline ─────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the quantum pipeline and return top molecules with scores."""
    print("Loading dataset...")
    df = fetch_dataset()
    print(f"Dataset: {len(df)} molecules")

    def fingerprint(smi):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp)

    X = np.array([fingerprint(s) for s in df.SMILES])
    y = df.Activity.values

    model, acc = train_classifier(X, y)
    print(f"Classical accuracy: {acc:.3f}")

    scores = model.predict_proba(X)[:, 1]
    all_active = np.where(y == 1)[0]
    all_inactive = np.where(y == 0)[0]
    active_ranked = all_active[np.argsort(scores[all_active])[::-1]]
    inactive_ranked = all_inactive[np.argsort(scores[all_inactive])]
    n_per_class = min(20, len(active_ranked), len(inactive_ranked))
    selected_idx = np.concatenate([active_ranked[:n_per_class], inactive_ranked[:n_per_class]])
    np.random.shuffle(selected_idx)

    X_sel = X[selected_idx]
    y_sel = y[selected_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    n_comp = min(3, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_quantum = pca.fit_transform(X_scaled)

    print("Running quantum stage...")
    q_scores = quantum_refinement(X_quantum, y_sel)
    print("Quantum stage completed")

    results = []
    for i, idx in enumerate(selected_idx):
        results.append({
            "smiles": df.iloc[idx]["SMILES"],
            "ic50": float(df.iloc[idx]["IC50"]),
            "activity": int(df.iloc[idx]["Activity"]),
            "quantum_score": float(q_scores[i]),
        })

    results.sort(key=lambda r: r["quantum_score"], reverse=True)
    return results[:5]


# ── Generate HTML ────────────────────────────────────────────────────────

def generate_html(molecules_data, output_path, reference_data=None, comparisons=None):
    """Write the self-contained HTML viewer with embedded molecule data."""
    molecules_json = json.dumps(molecules_data, indent=2)
    reference_json = json.dumps(reference_data or [], indent=2)
    comparisons_json = json.dumps(comparisons or {}, indent=2)
    html = _build_html(molecules_json, reference_json, comparisons_json)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Open in browser:  open {output_path}")


def _build_html(molecules_json, reference_json, comparisons_json):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Molecular Viewer</title>
<meta name="description" content="Interactive 3D quantum molecular visualization of top drug candidates">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Outfit',sans-serif;background:#06080f;color:#e0e4f0;overflow:hidden;height:100vh;width:100vw}}
#canvas-container{{position:fixed;top:0;left:0;width:100%;height:100%;z-index:1}}
canvas{{display:block}}
.glass{{background:rgba(12,18,35,0.65);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(0,229,255,0.12);border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.5),inset 0 0 30px rgba(0,229,255,0.03)}}
#title-bar{{position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:100;padding:12px 32px;display:flex;align-items:center;gap:12px}}
#title-bar h1{{font-size:18px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;background:linear-gradient(135deg,#00e5ff,#7c4dff,#ff3d5a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.quantum-dot{{width:8px;height:8px;border-radius:50%;background:#00e5ff;box-shadow:0 0 12px #00e5ff,0 0 30px rgba(0,229,255,0.3);animation:pulse-dot 2s ease-in-out infinite}}
@keyframes pulse-dot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:0.5;transform:scale(0.7)}}}}
#sidebar{{position:fixed;top:80px;left:20px;z-index:100;width:280px;padding:20px;max-height:calc(100vh - 100px);overflow-y:auto;transition:transform 0.3s ease,opacity 0.3s ease}}
#sidebar.collapsed{{transform:translateX(-310px);opacity:0;pointer-events:none}}
#sidebar h2{{font-size:12px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#00e5ff;margin-bottom:16px}}
.section-divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(255,215,0,0.3),transparent);margin:16px 0}}
.ref-heading{{font-size:12px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#ffd700;margin-bottom:12px}}
.molecule-card.reference{{border:1px solid rgba(255,215,0,0.15)}}
.molecule-card.reference:hover{{border-color:rgba(255,215,0,0.35);transform:translateX(4px)}}
.molecule-card.reference.active{{border-color:rgba(255,215,0,0.5);box-shadow:0 0 20px rgba(255,215,0,0.12)}}
.molecule-card.reference::before{{background:linear-gradient(135deg,rgba(255,215,0,0.06),transparent)}}
.molecule-card.reference .molecule-rank{{color:#ffd700}}
.ref-badge{{display:inline-block;font-size:8px;letter-spacing:1px;text-transform:uppercase;color:#ffd700;background:rgba(255,215,0,0.1);padding:2px 6px;border-radius:4px;margin-left:6px}}
.info-value.ref-highlight{{color:#ffd700;font-size:20px;font-weight:600}}
.molecule-card{{padding:14px;border-radius:12px;margin-bottom:8px;cursor:pointer;transition:all 0.3s ease;border:1px solid transparent;position:relative;overflow:hidden}}
.molecule-card::before{{content:'';position:absolute;top:0;left:0;width:100%;height:100%;background:linear-gradient(135deg,rgba(0,229,255,0.05),transparent);opacity:0;transition:opacity 0.3s}}
.molecule-card:hover::before,.molecule-card.active::before{{opacity:1}}
.molecule-card:hover{{border-color:rgba(0,229,255,0.2);transform:translateX(4px)}}
.molecule-card.active{{border-color:rgba(0,229,255,0.4);box-shadow:0 0 20px rgba(0,229,255,0.1)}}
.molecule-rank{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#00e5ff;font-weight:600;margin-bottom:4px}}
.molecule-name{{color:#c8cce0;font-family:'JetBrains Mono',monospace;word-break:break-all;line-height:1.4;font-size:10px;margin-bottom:8px;max-height:40px;overflow:hidden}}
.molecule-stats{{display:flex;gap:12px}}
.stat{{font-size:10px}}
.stat-label{{color:#6b7394;text-transform:uppercase;letter-spacing:1px;font-size:9px}}
.stat-value{{color:#e0e4f0;font-family:'JetBrains Mono',monospace;font-weight:400}}
.stat-value.active-1{{color:#4dff9e}}
.stat-value.active-0{{color:#ff6b6b}}
#info-panel{{position:fixed;bottom:20px;right:20px;z-index:100;width:320px;padding:24px}}
#info-panel h3{{font-size:12px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#7c4dff;margin-bottom:16px}}
.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.info-item{{display:flex;flex-direction:column;gap:4px}}
.info-item.full{{grid-column:1/-1}}
.info-label{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#6b7394}}
.info-value{{font-family:'JetBrains Mono',monospace;font-size:13px;color:#e0e4f0}}
.info-value.highlight{{color:#00e5ff;font-size:20px;font-weight:600}}
.info-smiles{{font-size:10px;word-break:break-all;line-height:1.5;color:#8b90a8;max-height:60px;overflow-y:auto}}
#controls{{position:fixed;bottom:20px;left:20px;z-index:100;padding:16px 20px;display:flex;gap:16px;align-items:center}}
.toggle{{display:flex;align-items:center;gap:8px;font-size:11px;color:#8b90a8;cursor:pointer;transition:color 0.3s}}
.toggle:hover{{color:#e0e4f0}}
.toggle-switch{{width:32px;height:18px;background:rgba(255,255,255,0.1);border-radius:9px;position:relative;transition:background 0.3s}}
.toggle-switch.on{{background:rgba(0,229,255,0.3)}}
.toggle-switch::after{{content:'';position:absolute;width:14px;height:14px;border-radius:50%;background:#6b7394;top:2px;left:2px;transition:all 0.3s}}
.toggle-switch.on::after{{left:16px;background:#00e5ff;box-shadow:0 0 8px rgba(0,229,255,0.5)}}
#element-legend{{position:fixed;top:80px;right:20px;z-index:100;padding:16px 20px}}
#element-legend h4{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#6b7394;margin-bottom:10px}}
.legend-items{{display:flex;flex-wrap:wrap;gap:8px}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:#8b90a8}}
.legend-dot{{width:10px;height:10px;border-radius:50%;box-shadow:0 0 6px currentColor}}
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:rgba(0,229,255,0.2);border-radius:2px}}
#loading{{position:fixed;top:0;left:0;width:100%;height:100%;background:#06080f;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:20px;z-index:1000;transition:opacity 0.8s ease}}
#loading.hide{{opacity:0;pointer-events:none}}
.loading-ring{{width:60px;height:60px;border:2px solid rgba(0,229,255,0.1);border-top-color:#00e5ff;border-radius:50%;animation:spin 1s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.loading-text{{font-size:12px;letter-spacing:3px;text-transform:uppercase;color:#6b7394}}
#compare-panel{{position:fixed;top:80px;right:20px;z-index:100;width:340px;padding:20px;display:none;transition:transform 0.3s ease,opacity 0.3s ease}}
#compare-panel.collapsed{{transform:translateX(370px);opacity:0;pointer-events:none}}
.panel-toggle{{position:fixed;z-index:200;width:36px;height:36px;border:1px solid rgba(0,229,255,0.25);border-radius:10px;background:rgba(12,18,35,0.8);backdrop-filter:blur(10px);color:#00e5ff;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.3s ease}}
.panel-toggle:hover{{background:rgba(0,229,255,0.15);border-color:rgba(0,229,255,0.5)}}
#sidebar-toggle{{top:80px;left:20px;display:none}}
#compare-toggle-btn{{top:80px;right:20px;display:none}}
#compare-panel h3{{font-size:12px;font-weight:500;letter-spacing:2px;text-transform:uppercase;background:linear-gradient(90deg,#4dff9e,#ffa94d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:14px}}
.compare-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)}}
.compare-row:last-child{{border-bottom:none}}
.compare-label{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#6b7394}}
.compare-value{{font-family:'JetBrains Mono',monospace;font-size:13px}}
.compare-summary{{font-size:11px;color:#c8cce0;line-height:1.6;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;margin-top:10px}}
.highlight-legend{{display:flex;gap:16px;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05)}}
.hl-item{{display:flex;align-items:center;gap:6px;font-size:10px;color:#8b90a8}}
.hl-dot{{width:10px;height:10px;border-radius:50%}}
.hl-dot.unique-cand{{background:linear-gradient(135deg,#00e5ff,#ff69b4,#4dff9e);box-shadow:0 0 8px #00e5ff}}
.hl-dot.unique-ref{{background:#ffa94d;box-shadow:0 0 8px #ffa94d}}
.hl-dot.common{{background:#556677;box-shadow:none;opacity:0.5}}
.ref-selector{{width:100%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,215,0,0.2);border-radius:8px;color:#e0e4f0;font-family:'Outfit',sans-serif;font-size:12px;padding:8px 12px;margin-bottom:14px;outline:none;cursor:pointer;appearance:none;-webkit-appearance:none}}
.ref-selector:focus{{border-color:rgba(255,215,0,0.5)}}
.ref-selector option{{background:#0c1223;color:#e0e4f0}}
.side-label{{position:absolute;top:-22px;font-size:10px;letter-spacing:2px;text-transform:uppercase;font-family:'JetBrains Mono',monospace}}
.side-label.left{{left:0;color:#ffa94d}}
.side-label.right{{right:0;color:#4dff9e}}
@keyframes pulse-unique{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:0.7;transform:scale(1.15)}}}}
</style>
</head>
<body>
<div id="loading"><div class="loading-ring"></div><div class="loading-text">Initializing Quantum Viewer</div></div>
<div id="title-bar" class="glass"><div class="quantum-dot"></div><h1>Quantum Molecular Viewer</h1><div class="quantum-dot"></div></div>
<div id="sidebar" class="glass"><h2>Top Candidates</h2><div id="molecule-list"></div><div class="section-divider"></div><div class="ref-heading">Reference Drugs &#x1f3af;</div><div id="reference-list"></div></div>
<div id="compare-panel" class="glass">
    <h3>&#9889; Structural Comparison</h3>
    <select id="ref-selector" class="ref-selector" onchange="onRefSelectorChange()"></select>
    <div class="compare-row"><span class="compare-label">Common Atoms (MCS)</span><span class="compare-value" id="cmp-common">&mdash;</span></div>
    <div class="compare-row"><span class="compare-label">Unique to Candidate</span><span class="compare-value" style="color:#00e5ff" id="cmp-cand-unique">&mdash;</span></div>
    <div class="compare-row"><span class="compare-label">Unique to Reference</span><span class="compare-value" style="color:#ffa94d" id="cmp-ref-unique">&mdash;</span></div>
    <div class="compare-summary" id="cmp-summary">&mdash;</div>
    <div class="highlight-legend">
        <div class="hl-item"><div class="hl-dot unique-cand"></div>Unique (vivid)</div>
        <div class="hl-item"><div class="hl-dot common"></div>Common (faded)</div>
    </div>
</div>
<div id="info-panel" class="glass">
    <h3>Molecule Details</h3>
    <div class="info-grid">
        <div class="info-item"><span class="info-label">Quantum Score</span><span class="info-value highlight" id="info-qscore">&mdash;</span></div>
        <div class="info-item"><span class="info-label">Type</span><span class="info-value" id="info-type">&mdash;</span></div>
        <div class="info-item"><span class="info-label">Activity</span><span class="info-value" id="info-activity">&mdash;</span></div>
        <div class="info-item"><span class="info-label">IC50 (nM)</span><span class="info-value" id="info-ic50">&mdash;</span></div>
        <div class="info-item"><span class="info-label">Atoms</span><span class="info-value" id="info-atoms">&mdash;</span></div>
        <div class="info-item full"><span class="info-label">SMILES</span><span class="info-value info-smiles" id="info-smiles">&mdash;</span></div>
        <div class="info-item full" id="info-desc-row" style="display:none"><span class="info-label">Description</span><span class="info-value" id="info-desc" style="font-size:11px;color:#ffd700">&mdash;</span></div>
    </div>
</div>
<div id="controls" class="glass">
    <div class="toggle" onclick="toggleAutoRotate()"><div class="toggle-switch on" id="toggle-rotate"></div><span>Auto-Rotate</span></div>
    <div class="toggle" onclick="toggleLabels()"><div class="toggle-switch" id="toggle-labels"></div><span>Labels</span></div>
    <div class="toggle" onclick="toggleParticles()"><div class="toggle-switch on" id="toggle-particles"></div><span>Quantum Field</span></div>
    <div class="toggle" onclick="toggleCompare()"><div class="toggle-switch" id="toggle-compare"></div><span>Compare &#9889;</span></div>
</div>
<div id="element-legend" class="glass"><h4>Elements</h4><div class="legend-items" id="legend-items"></div></div>
<button id="sidebar-toggle" class="panel-toggle" onclick="toggleSidebar()" title="Toggle Candidates">&#9776;</button>
<button id="compare-toggle-btn" class="panel-toggle" onclick="toggleComparePanel()" title="Toggle Comparison">&#9881;</button>
<div id="canvas-container"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.146.0/examples/js/controls/OrbitControls.js"></script>
<script>
const MOLECULES = {molecules_json};
const REFERENCES = {reference_json};
const COMPARISONS = {comparisons_json};
let currentMolIndex = -1, currentIsRef = false, autoRotate = true, showLabels = false, showParticles = true;
let compareMode = false, compareRefIndex = 0;
let moleculeGroup = null, refMolGroup = null, labelSprites = [], refLabelSprites = [], particleSystem = null;

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 20);
const renderer = new THREE.WebGLRenderer({{ antialias:true, alpha:true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputEncoding = THREE.sRGBEncoding;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;
controls.minDistance = 5;
controls.maxDistance = 50;

scene.add(new THREE.AmbientLight(0x1a1a2e, 0.8));
const pl1 = new THREE.PointLight(0x00e5ff, 2, 80); pl1.position.set(15,15,15); scene.add(pl1);
const pl2 = new THREE.PointLight(0x7c4dff, 1.5, 80); pl2.position.set(-15,-10,10); scene.add(pl2);
const pl3 = new THREE.PointLight(0xff3d5a, 0.8, 80); pl3.position.set(0,-15,-15); scene.add(pl3);

function createBgParticles() {{
    const N=2000, pos=new Float32Array(N*3), col=new Float32Array(N*3);
    for(let i=0;i<N;i++){{
        pos[i*3]=(Math.random()-0.5)*100;pos[i*3+1]=(Math.random()-0.5)*100;pos[i*3+2]=(Math.random()-0.5)*100;
        const c=new THREE.Color().setHSL(0.52+Math.random()*0.15,0.8,0.4+Math.random()*0.3);
        col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;
    }}
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.BufferAttribute(pos,3));
    g.setAttribute('color',new THREE.BufferAttribute(col,3));
    return new THREE.Points(g, new THREE.PointsMaterial({{size:0.08,vertexColors:true,transparent:true,opacity:0.4,blending:THREE.AdditiveBlending,depthWrite:false}}));
}}
const bgParticles = createBgParticles(); scene.add(bgParticles);

function createQuantumCloud(radius) {{
    const N=500, pos=new Float32Array(N*3), col=new Float32Array(N*3), vels=[];
    for(let i=0;i<N;i++){{
        const th=Math.random()*Math.PI*2, ph=Math.acos(2*Math.random()-1), r=radius*(0.8+Math.random()*0.8);
        pos[i*3]=r*Math.sin(ph)*Math.cos(th);pos[i*3+1]=r*Math.sin(ph)*Math.sin(th);pos[i*3+2]=r*Math.cos(ph);
        const c=new THREE.Color().setHSL(0.52+Math.random()*0.2,0.7,0.5+Math.random()*0.3);
        col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;
        vels.push({{vx:(Math.random()-0.5)*0.01,vy:(Math.random()-0.5)*0.01,vz:(Math.random()-0.5)*0.01}});
    }}
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.BufferAttribute(pos,3));
    g.setAttribute('color',new THREE.BufferAttribute(col,3));
    const p=new THREE.Points(g, new THREE.PointsMaterial({{size:0.06,vertexColors:true,transparent:true,opacity:0.35,blending:THREE.AdditiveBlending,depthWrite:false}}));
    p.userData.vels=vels; p.userData.radius=radius; return p;
}}

function buildMolecule(data, highlightIndices, highlightColor, dimMode) {{
    const group=new THREE.Group(), sprites=[];
    let cx=0,cy=0,cz=0,n=0;
    data.atoms.forEach(a=>{{if(a.element!=='H'){{cx+=a.x;cy+=a.y;cz+=a.z;n++}}}});
    if(n===0) data.atoms.forEach(a=>{{cx+=a.x;cy+=a.y;cz+=a.z;n++}});
    cx/=n;cy/=n;cz/=n;
    const hlSet=new Set(highlightIndices||[]);
    const positions=[];
    let heavyIdx=-1;
    data.atoms.forEach((atom,i)=>{{
        const pos=new THREE.Vector3(atom.x-cx,atom.y-cy,atom.z-cz);
        positions.push(pos);
        if(atom.element==='H') return;
        heavyIdx++;
        const isHighlighted=hlSet.has(heavyIdx);
        const isDimmed=dimMode&&!isHighlighted;
        // Unique atoms: vivid element colors. Common atoms: grey transparent.
        const color=isDimmed?new THREE.Color('#556677'):new THREE.Color(atom.color);
        const emissiveI=isDimmed?0.1:0.6;
        const opacity=isDimmed?0.3:1.0;
        const radius=atom.radius;
        const sphere=new THREE.Mesh(new THREE.SphereGeometry(radius,32,32),
            new THREE.MeshStandardMaterial({{color,emissive:color,emissiveIntensity:emissiveI,metalness:0.3,roughness:0.3,transparent:true,opacity}}));
        sphere.position.copy(pos); group.add(sphere);
        const glowOpacity=isDimmed?0.0:0.06;
        const glow=new THREE.Mesh(new THREE.SphereGeometry(atom.radius*2.0,16,16),
            new THREE.MeshBasicMaterial({{color,transparent:true,opacity:glowOpacity,blending:THREE.AdditiveBlending,depthWrite:false}}));
        glow.position.copy(pos); group.add(glow);
        const canvas=document.createElement('canvas');canvas.width=64;canvas.height=32;
        const ctx=canvas.getContext('2d');ctx.font='bold 20px Outfit';
        ctx.fillStyle=isDimmed?'#556677':atom.color;
        ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(atom.element,32,16);
        const sprite=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(canvas),transparent:true,depthWrite:false,opacity:0}}));
        sprite.position.copy(pos);sprite.position.y+=atom.radius+0.4;sprite.scale.set(1.2,0.6,1);
        group.add(sprite);sprites.push(sprite);
    }});
    data.bonds.forEach(bond=>{{
        const sa=data.atoms[bond.start],ea=data.atoms[bond.end];
        if(sa.element==='H'||ea.element==='H') return;
        const start=positions[bond.start],end=positions[bond.end];
        if(!start||!end) return;
        const dir=new THREE.Vector3().subVectors(end,start),len=dir.length();
        const mid=new THREE.Vector3().addVectors(start,end).multiplyScalar(0.5);
        const bondColor=dimMode?new THREE.Color('#333'):new THREE.Color().lerpColors(new THREE.Color(sa.color),new THREE.Color(ea.color),0.5);
        const bondOpacity=dimMode?0.3:0.75;
        const offsets=bond.order===1?[0]:bond.order===2?[-0.08,0.08]:[-0.12,0,0.12];
        offsets.forEach(offset=>{{
            const cyl=new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,len,8),
                new THREE.MeshStandardMaterial({{color:bondColor,emissive:bondColor,emissiveIntensity:0.4,metalness:0.5,roughness:0.3,transparent:true,opacity:bondOpacity}}));
            cyl.position.copy(mid);
            if(offset!==0){{const perp=new THREE.Vector3().crossVectors(dir,new THREE.Vector3(0,1,0)).normalize();
            if(perp.length()<0.01)perp.crossVectors(dir,new THREE.Vector3(1,0,0)).normalize();
            cyl.position.add(perp.multiplyScalar(offset));}}
            cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.clone().normalize());
            group.add(cyl);
        }});
    }});
    let maxD=0;
    data.atoms.forEach(a=>{{if(a.element==='H')return;const d=Math.sqrt((a.x-cx)**2+(a.y-cy)**2+(a.z-cz)**2);if(d>maxD)maxD=d}});
    return {{group,sprites,cloudRadius:maxD+2}};
}}

function populateUI() {{
    const list=document.getElementById('molecule-list');list.innerHTML='';
    MOLECULES.forEach((mol,i)=>{{
        const card=document.createElement('div');
        card.className='molecule-card'+(i===0?' active':'');card.id='mol-card-'+i;
        card.onclick=()=>selectMolecule(i,false);
        card.innerHTML=`<div class="molecule-rank">#${{i+1}}</div><div class="molecule-name">${{mol.smiles}}</div>
        <div class="molecule-stats"><div class="stat"><div class="stat-label">Q-Score</div><div class="stat-value">${{mol.quantum_score.toFixed(3)}}</div></div>
        <div class="stat"><div class="stat-label">IC50</div><div class="stat-value">${{mol.ic50.toFixed(1)}}</div></div>
        <div class="stat"><div class="stat-label">Active</div><div class="stat-value active-${{mol.activity}}">${{mol.activity?'Yes':'No'}}</div></div></div>`;
        list.appendChild(card);
    }});
    const refList=document.getElementById('reference-list');refList.innerHTML='';
    REFERENCES.forEach((ref,i)=>{{
        const card=document.createElement('div');
        card.className='molecule-card reference';card.id='ref-card-'+i;
        card.onclick=()=>selectMolecule(i,true);
        card.innerHTML=`<div class="molecule-rank">${{ref.name}} <span class="ref-badge">Approved Drug</span></div><div class="molecule-name">${{ref.smiles}}</div>
        <div class="molecule-stats"><div class="stat"><div class="stat-label">IC50</div><div class="stat-value">${{ref.ic50.toFixed(1)}} nM</div></div>
        <div class="stat"><div class="stat-label">Active</div><div class="stat-value active-1">Yes</div></div></div>`;
        refList.appendChild(card);
    }});
    const legendEl=document.getElementById('legend-items');const elements={{}};
    MOLECULES.concat(REFERENCES).forEach(mol=>{{mol.structure.atoms.forEach(a=>{{if(a.element!=='H')elements[a.element]=a.color}})}});
    legendEl.innerHTML=Object.entries(elements).map(([el,color])=>
        `<div class="legend-item"><div class="legend-dot" style="background:${{color}};color:${{color}}"></div>${{el}}</div>`).join('');
}}

function updateInfoPanel(mol, isRef) {{
    const qEl=document.getElementById('info-qscore');
    const typeEl=document.getElementById('info-type');
    const descRow=document.getElementById('info-desc-row');
    const descEl=document.getElementById('info-desc');
    if(isRef){{
        qEl.textContent='N/A';
        qEl.className='info-value ref-highlight';
        typeEl.textContent='Reference Drug';
        typeEl.style.color='#ffd700';
        descRow.style.display='';
        descEl.textContent=mol.description||'';
    }}else{{
        qEl.textContent=mol.quantum_score.toFixed(4);
        qEl.className='info-value highlight';
        typeEl.textContent='Pipeline Candidate';
        typeEl.style.color='#00e5ff';
        descRow.style.display='none';
    }}
    document.getElementById('info-activity').textContent='Active';
    document.getElementById('info-activity').style.color='#4dff9e';
    document.getElementById('info-ic50').textContent=mol.ic50.toFixed(2);
    document.getElementById('info-atoms').textContent=mol.structure.atoms.filter(a=>a.element!=='H').length;
    document.getElementById('info-smiles').textContent=mol.smiles;
}}

function clearScene(){{
    if(moleculeGroup){{scene.remove(moleculeGroup);moleculeGroup=null}}
    if(refMolGroup){{scene.remove(refMolGroup);refMolGroup=null}}
    if(particleSystem){{scene.remove(particleSystem);particleSystem=null}}
    labelSprites=[];refLabelSprites=[];
}}

function createSideLabel(text, color, xPos, yPos){{
    const canvas=document.createElement('canvas');canvas.width=512;canvas.height=64;
    const ctx=canvas.getContext('2d');
    ctx.font='bold 36px Outfit';ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.shadowColor=color;ctx.shadowBlur=12;
    ctx.fillText(text,256,32);
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(canvas),transparent:true,depthWrite:false}}));
    sprite.position.set(xPos,yPos,0);sprite.scale.set(6,0.75,1);
    return sprite;
}}

function renderCompareView(){{
    clearScene();
    if(currentIsRef||currentMolIndex<0) return;
    const ci=currentMolIndex, ri=compareRefIndex;
    const key=ci+'-'+ri;
    const cmp=COMPARISONS[key]||{{cand_unique:[],ref_unique:[],common_count:0,cand_unique_count:0,ref_unique_count:0,summary:'N/A'}};
    const cand=MOLECULES[ci];
    // Show single candidate molecule: unique atoms in vivid colors, common atoms grey
    const candResult=buildMolecule(cand.structure,cmp.cand_unique,'#4dff9e',true);
    scene.add(candResult.group);moleculeGroup=candResult.group;labelSprites=candResult.sprites;
    // Auto-enable labels
    showLabels=true;document.getElementById('toggle-labels').classList.add('on');
    labelSprites.forEach(s=>{{s.material.opacity=1}});
    controls.autoRotate=autoRotate;
    // Update compare panel
    document.getElementById('cmp-common').textContent=cmp.common_count+' atoms';
    document.getElementById('cmp-cand-unique').textContent=cmp.cand_unique_count+' atoms';
    document.getElementById('cmp-ref-unique').textContent=cmp.ref_unique_count+' atoms';
    document.getElementById('cmp-summary').textContent=cmp.summary;
    camera.position.setLength(Math.max(candResult.cloudRadius*3,14));
}}

function renderNormalView(){{
    clearScene();
    const mol=currentIsRef?REFERENCES[currentMolIndex]:MOLECULES[currentMolIndex];
    if(!mol) return;
    const {{group,sprites,cloudRadius}}=buildMolecule(mol.structure);
    scene.add(group);moleculeGroup=group;labelSprites=sprites;
    particleSystem=createQuantumCloud(cloudRadius);particleSystem.visible=showParticles;scene.add(particleSystem);
    camera.position.setLength(Math.max(cloudRadius*3,12));
}}

function selectMolecule(index, isRef) {{
    if(index===currentMolIndex && isRef===currentIsRef) return;
    document.querySelectorAll('.molecule-card').forEach(c=>c.classList.remove('active'));
    const cardId=isRef?'ref-card-'+index:'mol-card-'+index;
    const card=document.getElementById(cardId);if(card)card.classList.add('active');
    currentMolIndex=index;currentIsRef=isRef;
    const mol=isRef?REFERENCES[index]:MOLECULES[index];
    updateInfoPanel(mol,isRef);
    if(compareMode&&!isRef){{renderCompareView()}}
    else{{if(compareMode){{compareMode=false;document.getElementById('toggle-compare').classList.remove('on');document.getElementById('compare-panel').style.display='none';document.getElementById('info-panel').style.display='';document.getElementById('element-legend').style.display=''}}
    renderNormalView();}}
}}

function toggleAutoRotate(){{autoRotate=!autoRotate;controls.autoRotate=autoRotate;document.getElementById('toggle-rotate').classList.toggle('on',autoRotate)}}
function toggleLabels(){{showLabels=!showLabels;document.getElementById('toggle-labels').classList.toggle('on',showLabels);
    labelSprites.forEach(s=>{{s.material.opacity=showLabels?1:0}});
    refLabelSprites.forEach(s=>{{s.material.opacity=showLabels?1:0}});
}}
function toggleParticles(){{showParticles=!showParticles;document.getElementById('toggle-particles').classList.toggle('on',showParticles);if(particleSystem)particleSystem.visible=showParticles}}

function toggleCompare(){{
    if(currentIsRef) return;
    compareMode=!compareMode;
    document.getElementById('toggle-compare').classList.toggle('on',compareMode);
    document.getElementById('compare-panel').style.display=compareMode?'block':'none';
    document.getElementById('info-panel').style.display=compareMode?'none':'';
    document.getElementById('element-legend').style.display=compareMode?'none':'';
    // Show toggle buttons when in compare mode, hide sidebar for more space
    document.getElementById('sidebar-toggle').style.display=compareMode?'flex':'none';
    document.getElementById('compare-toggle-btn').style.display=compareMode?'flex':'none';
    if(compareMode){{
        document.getElementById('sidebar').classList.add('collapsed');
        sidebarVisible=false;
    }}else{{
        document.getElementById('sidebar').classList.remove('collapsed');
        document.getElementById('compare-panel').classList.remove('collapsed');
        sidebarVisible=true;comparePanelVisible=true;
        controls.autoRotate=autoRotate;
    }}
    if(compareMode){{renderCompareView()}}
    else{{renderNormalView();updateInfoPanel(MOLECULES[currentMolIndex],false)}}
}}
let sidebarVisible=true, comparePanelVisible=true;
function toggleSidebar(){{
    sidebarVisible=!sidebarVisible;
    document.getElementById('sidebar').classList.toggle('collapsed',!sidebarVisible);
}}
function toggleComparePanel(){{
    comparePanelVisible=!comparePanelVisible;
    document.getElementById('compare-panel').classList.toggle('collapsed',!comparePanelVisible);
}}

function onRefSelectorChange(){{
    compareRefIndex=parseInt(document.getElementById('ref-selector').value)||0;
    if(compareMode) renderCompareView();
}}

const clock=new THREE.Clock();
function animate(){{
    requestAnimationFrame(animate);
    const t=clock.getElapsedTime();
    controls.update();
    if(bgParticles){{bgParticles.rotation.y=t*0.02;bgParticles.rotation.x=Math.sin(t*0.01)*0.1}}
    if(particleSystem&&particleSystem.visible){{
        const pos=particleSystem.geometry.attributes.position.array;
        const vels=particleSystem.userData.vels,R=particleSystem.userData.radius;
        for(let i=0;i<vels.length;i++){{
            pos[i*3]+=vels[i].vx+Math.sin(t*2+i)*0.003;
            pos[i*3+1]+=vels[i].vy+Math.cos(t*2+i)*0.003;
            pos[i*3+2]+=vels[i].vz+Math.sin(t*3+i*0.5)*0.003;
            const d=Math.sqrt(pos[i*3]**2+pos[i*3+1]**2+pos[i*3+2]**2);
            if(d>R*1.5){{pos[i*3]*=0.5;pos[i*3+1]*=0.5;pos[i*3+2]*=0.5}}
        }}
        particleSystem.geometry.attributes.position.needsUpdate=true;
    }}
    pl1.position.x=Math.sin(t*0.3)*15;pl1.position.z=Math.cos(t*0.3)*15;
    pl2.position.y=Math.sin(t*0.2)*12;
    renderer.render(scene,camera);
}}

window.addEventListener('resize',()=>{{
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
}});

// Populate reference selector dropdown
const refSel=document.getElementById('ref-selector');
REFERENCES.forEach((r,i)=>{{const opt=document.createElement('option');opt.value=i;opt.textContent=r.name;refSel.appendChild(opt)}});
populateUI();selectMolecule(0,false);animate();
setTimeout(()=>{{document.getElementById('loading').classList.add('hide')}},800);
</script>
</body>
</html>'''


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Quantum Molecular Visualizer")
    print("=" * 60)
    print()

    top_molecules = run_pipeline()

    print("\nGenerating 3D structures for pipeline candidates...")
    molecules_data = []
    for i, mol in enumerate(top_molecules):
        print(f"  [{i+1}/{len(top_molecules)}] {mol['smiles'][:50]}...")
        structure = mol_to_3d_data(mol["smiles"])
        if structure:
            molecules_data.append({
                "smiles": mol["smiles"],
                "ic50": mol["ic50"],
                "activity": mol["activity"],
                "quantum_score": mol["quantum_score"],
                "structure": structure,
            })

    if not molecules_data:
        print("ERROR: No valid 3D structures generated.")
        exit(1)

    print(f"\n✓ Generated 3D data for {len(molecules_data)} pipeline candidates")

    # Generate 3D structures for reference (known) DPP-4 inhibitors
    print("\nGenerating 3D structures for reference drugs...")
    reference_data = []
    for i, ref in enumerate(REFERENCE_MOLECULES):
        print(f"  [{i+1}/{len(REFERENCE_MOLECULES)}] {ref['name']}...")
        structure = mol_to_3d_data(ref["smiles"])
        if structure:
            reference_data.append({
                "name": ref["name"],
                "smiles": ref["smiles"],
                "ic50": ref["ic50"],
                "activity": 1,
                "description": ref["description"],
                "structure": structure,
            })

    print(f"✓ Generated 3D data for {len(reference_data)} reference drugs")

    # Compute MCS-based structural comparisons
    print("\nComputing structural comparisons (MCS)...")
    comparisons = compute_comparison_data(molecules_data, reference_data)
    print(f"✓ Computed {len(comparisons)} comparison pairs")

    output_path = os.path.join(os.path.dirname(__file__), "molecule_viewer.html")
    generate_html(molecules_data, output_path, reference_data=reference_data, comparisons=comparisons)

    print()
    print("=" * 60)
    print("  Done! Open molecule_viewer.html in your browser.")
    print("=" * 60)
