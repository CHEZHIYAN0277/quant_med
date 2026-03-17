"""
Structure-Based Validation Module
=================================
Extends the hybrid drug-discovery pipeline with molecular docking using
AutoDock Vina to validate top candidate molecules against the DPP-4 target
protein (PDB: 4A5S).

Steps:
  1. Read candidate molecules from CSV
  2. Generate 3D structures (RDKit)
  3. Convert ligands to PDBQT (Meeko)
  4. Prepare protein structure (PDBFixer + Meeko)
  5. Dock each ligand with AutoDock Vina
  6. Rank by binding energy
  7. Save results to CSV
  8. (Optional) Visualise with py3Dmol

Installation (inside your venv):
    pip install meeko vina pdbfixer py3Dmol

Usage:
    python structure_based_validation.py
"""

import os
import sys
import warnings
import tempfile
import subprocess
import textwrap

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = "top_molecules_for_docking.csv"
OUTPUT_CSV = "docking_results.csv"
OUTPUT_DIR = "output"                       # intermediate .sdf / .pdbqt files
PDB_ID = "4A5S"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"

# DPP-4 active-site box (Å) — derived from the co-crystallised ligand in 4A5S
# These coordinates centre on the catalytic pocket.
BOX_CENTER = (38.0, 48.0, 37.0)            # (x, y, z) in Å
BOX_SIZE = (25.0, 25.0, 25.0)              # generous box to cover the pocket

VINA_EXHAUSTIVENESS = 8                     # docking thoroughness (default)
VINA_N_POSES = 5                            # number of binding modes to return


# ===========================================================================
# STEP 1 — Read input CSV
# ===========================================================================

def load_candidates(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV produced by the upstream pipeline.
    Expected columns: SMILES, IC50, Activity
    """
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"SMILES", "IC50", "Activity"}
    if not required.issubset(df.columns):
        sys.exit(f"[ERROR] CSV must contain columns: {required}")

    print(f"[STEP 1] Loaded {len(df)} candidate molecule(s) from {csv_path}")
    return df


# ===========================================================================
# STEP 2 — Generate 3D structures and save as .sdf
# ===========================================================================

def generate_3d_sdf(smiles: str, sdf_path: str) -> bool:
    """
    SMILES → RDKit mol → add Hs → embed 3D → UFF optimise → write .sdf.
    Returns True on success.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"Could not parse SMILES: {smiles}")
        return False

    # Add explicit hydrogens (required for docking)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates with the ETKDG method
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Fallback: try without the distance-geometry parameter set
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            warnings.warn(f"3D embedding failed for: {smiles}")
            return False

    # Geometry optimisation with the Universal Force Field
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        warnings.warn(f"UFF optimisation failed for: {smiles}; using unoptimised geometry")

    # Write .sdf
    writer = rdmolfiles.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return True


def prepare_ligand_sdfs(df: pd.DataFrame, out_dir: str) -> list:
    """
    Generate .sdf files for every candidate and return a list of dicts with
    metadata + file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for idx, row in df.iterrows():
        sdf_path = os.path.join(out_dir, f"ligand_{idx}.sdf")
        ok = generate_3d_sdf(row["SMILES"], sdf_path)
        if ok:
            records.append({
                "index": idx,
                "SMILES": row["SMILES"],
                "IC50": row["IC50"],
                "Activity": row["Activity"],
                "sdf_path": sdf_path,
            })
        else:
            print(f"  ⚠ Skipping molecule {idx} (3D generation failed)")

    print(f"[STEP 2] Generated 3D structures for {len(records)}/{len(df)} molecules")
    return records


# ===========================================================================
# STEP 3 — Convert .sdf → .pdbqt (Meeko)
# ===========================================================================

def sdf_to_pdbqt(sdf_path: str, pdbqt_path: str) -> bool:
    """
    Use Meeko's MoleculePreparation to convert an SDF file into PDBQT format
    suitable for AutoDock Vina.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    # Read the SDF with RDKit (Meeko operates on RDKit mol objects)
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(supplier, None)
    if mol is None:
        warnings.warn(f"Could not read SDF: {sdf_path}")
        return False

    # Meeko preparation — detects rotatable bonds, assigns atom types & charges
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    # Write PDBQT
    for setup in mol_setups:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(pdbqt_path, "w") as fh:
                fh.write(pdbqt_string)
            return True
        else:
            warnings.warn(f"Meeko PDBQT write error: {error_msg}")
            return False

    return False


def prepare_ligand_pdbqts(records: list, out_dir: str) -> list:
    """
    Convert every .sdf → .pdbqt and update records in-place.
    Returns only the records where conversion succeeded.
    """
    good = []
    for rec in records:
        pdbqt_path = os.path.join(out_dir, f"ligand_{rec['index']}.pdbqt")
        if sdf_to_pdbqt(rec["sdf_path"], pdbqt_path):
            rec["pdbqt_path"] = pdbqt_path
            good.append(rec)
        else:
            print(f"  ⚠ PDBQT conversion failed for molecule {rec['index']}")

    print(f"[STEP 3] Converted {len(good)}/{len(records)} ligands to PDBQT")
    return good


# ===========================================================================
# STEP 4 — Protein preparation
# ===========================================================================

def download_pdb(pdb_id: str, dest_path: str):
    """Download a PDB file from the RCSB."""
    import requests

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "w") as fh:
        fh.write(resp.text)
    print(f"  Saved raw PDB to {dest_path}")


def prepare_protein(out_dir: str) -> str:
    """
    Download PDB 4A5S, remove water & heteroatoms, add hydrogens, and convert
    to PDBQT.

    Uses PDBFixer (from OpenMM) for robust preparation, then Meeko for the
    PDBQT conversion of the receptor.

    Returns the path to the prepared protein PDBQT file.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    os.makedirs(out_dir, exist_ok=True)

    raw_pdb = os.path.join(out_dir, f"{PDB_ID}_raw.pdb")
    clean_pdb = os.path.join(out_dir, f"{PDB_ID}_clean.pdb")
    protein_pdbqt = os.path.join(out_dir, f"{PDB_ID}_protein.pdbqt")

    # Download if not already cached
    if not os.path.isfile(raw_pdb):
        download_pdb(PDB_ID, raw_pdb)

    # --- PDBFixer: clean the structure ---
    print("  Cleaning protein structure with PDBFixer …")
    fixer = PDBFixer(filename=raw_pdb)

    # Remove water molecules
    fixer.removeHeterogens(keepWater=False)

    # Find and add missing residues / atoms (non-terminal only)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Add hydrogens at pH 7.4
    fixer.addMissingHydrogens(pH=7.4)

    # Write the cleaned PDB
    with open(clean_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    print(f"  Cleaned PDB saved to {clean_pdb}")

    # --- Convert to PDBQT for Vina ---
    # For the receptor we use a simple approach: write an ATOM-only PDB and
    # use the `prepare_receptor` helper from Meeko / AutoDockTools-py or
    # fall back to a minimal conversion.
    _convert_protein_to_pdbqt(clean_pdb, protein_pdbqt)

    print(f"[STEP 4] Protein prepared → {protein_pdbqt}")
    return protein_pdbqt


def _convert_protein_to_pdbqt(pdb_path: str, pdbqt_path: str):
    """
    Convert a cleaned protein PDB to PDBQT format.

    Strategy: Try using the `prepare_receptor` script from
    ADFR-suite / meeko-utils.  If unavailable, perform a minimal conversion
    that assigns Vina-compatible atom types (sufficient for docking).
    """
    # Try the ADFR-suite command-line tool first
    try:
        result = subprocess.run(
            ["prepare_receptor", "-r", pdb_path, "-o", pdbqt_path, "-A", "hydrogens"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(pdbqt_path):
            print("  Used ADFR prepare_receptor for PDBQT conversion.")
            return
    except FileNotFoundError:
        pass  # prepare_receptor not installed — fall through

    # Fallback: minimal PDB → PDBQT conversion
    # AutoDock Vina is fairly tolerant; the main addition is the atom-type
    # column in PDBQT format.  We map common elements → AD4 atom types.
    ad4_types = {
        "C": "C", "N": "N", "O": "OA", "S": "SA",
        "H": "HD", "F": "F", "P": "P", "CL": "Cl",
        "BR": "Br", "I": "I", "ZN": "Zn", "FE": "Fe",
        "MG": "Mg", "CA": "Ca", "MN": "Mn",
    }

    print("  Using built-in minimal PDB → PDBQT converter …")
    with open(pdb_path) as fin, open(pdbqt_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if not element:
                    # Guess from atom name
                    atom_name = line[12:16].strip()
                    element = atom_name[0] if atom_name else "C"
                ad_type = ad4_types.get(element, element)
                # PDBQT is PDB + partial charge (cols 71-76) + atom type (cols 77-79)
                pdb_line = line[:54]
                # Pad to column 70, add dummy partial charge and atom type
                pdb_line = f"{pdb_line:<70s}"
                pdb_line = pdb_line[:70] + f"  0.000 {ad_type:<2s}\n"
                fout.write(pdb_line)
            elif line.startswith(("TER", "END")):
                fout.write(line)
    print(f"  Minimal PDBQT written to {pdbqt_path}")


# ===========================================================================
# STEP 5 — Molecular docking with AutoDock Vina
# ===========================================================================

def dock_ligand(protein_pdbqt: str, ligand_pdbqt: str,
                center: tuple, box: tuple,
                exhaustiveness: int = VINA_EXHAUSTIVENESS,
                n_poses: int = VINA_N_POSES) -> float:
    """
    Run AutoDock Vina for a single ligand and return the best binding
    affinity (kcal/mol).  Returns np.nan on failure.
    """
    from vina import Vina

    v = Vina(sf_name="vina")
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)

    v.compute_vina_maps(center=list(center), box_size=list(box))

    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    # energies() returns a list of [affinity, ...] for each pose
    energies = v.energies()
    if energies is not None and len(energies) > 0:
        best_affinity = energies[0][0]   # kcal/mol, most negative = best
        return best_affinity

    return np.nan


def run_docking(records: list, protein_pdbqt: str) -> list:
    """
    Dock every ligand against the prepared protein.
    Appends 'docking_score' to each record.
    """
    total = len(records)
    for i, rec in enumerate(records, 1):
        smiles_short = rec["SMILES"][:40] + ("…" if len(rec["SMILES"]) > 40 else "")
        print(f"  [{i}/{total}] Docking: {smiles_short}")
        score = dock_ligand(protein_pdbqt, rec["pdbqt_path"],
                            center=BOX_CENTER, box=BOX_SIZE)
        rec["docking_score"] = score
        print(f"           → {score:.2f} kcal/mol" if not np.isnan(score)
              else "           → FAILED")

    print(f"[STEP 5] Docking complete for {total} molecule(s)")
    return records


# ===========================================================================
# STEP 6 — Ranking
# ===========================================================================

def rank_results(records: list) -> pd.DataFrame:
    """
    Build a DataFrame sorted by best binding energy (most negative first).
    """
    df = pd.DataFrame(records)
    df = df.rename(columns={"docking_score": "docking_score_kcal_mol"})
    df = df.sort_values("docking_score_kcal_mol", ascending=True)
    df = df.reset_index(drop=True)

    # Keep only the columns the user requested
    result_cols = ["SMILES", "IC50", "docking_score_kcal_mol"]
    df_out = df[result_cols]

    print("[STEP 6] Results ranked by binding energy (ascending = best)")
    print(df_out.to_string(index=False))
    return df_out


# ===========================================================================
# STEP 7 — Save results
# ===========================================================================

def save_results(df: pd.DataFrame, csv_path: str):
    """Write the ranked results to CSV."""
    df.to_csv(csv_path, index=False)
    print(f"[STEP 7] Results saved to {csv_path}")


# ===========================================================================
# STEP 8 — Visualization (py3Dmol)
# ===========================================================================

def visualise_docking(protein_pdb_path: str, ligand_sdf_path: str):
    """
    Render an interactive 3D view of the protein with the docked ligand.
    Works in Jupyter notebooks.  When run from a script, prints instructions
    for launching the viewer in a notebook instead.
    """
    try:
        import py3Dmol
    except ImportError:
        print("[STEP 8] py3Dmol not installed — skipping visualisation.")
        print("         Install with:  pip install py3Dmol")
        return None

    # Check if we are inside a Jupyter environment
    try:
        get_ipython()  # noqa: F821 — only defined in IPython / Jupyter
        in_notebook = True
    except NameError:
        in_notebook = False

    if not in_notebook:
        print("[STEP 8] Visualisation is best viewed in a Jupyter notebook.")
        print("         Launch a notebook and run:")
        print()
        print(textwrap.dedent(f"""\
            from structure_based_validation import visualise_docking
            view = visualise_docking(
                "{protein_pdb_path}",
                "{ligand_sdf_path}",
            )
            view.show()
        """))
        return None

    # --- Build the 3D viewer ---
    viewer = py3Dmol.view(width=800, height=600)

    # Load protein
    with open(protein_pdb_path) as fh:
        protein_data = fh.read()
    viewer.addModel(protein_data, "pdb")
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum", "opacity": 0.75}})

    # Show binding-site surface around the box centre
    viewer.addSurface(
        py3Dmol.VDW,
        {"opacity": 0.15, "color": "white"},
        {"model": 0, "within": {"distance": 12, "sel": {"x": BOX_CENTER[0],
                                                          "y": BOX_CENTER[1],
                                                          "z": BOX_CENTER[2]}}},
    )

    # Load docked ligand
    with open(ligand_sdf_path) as fh:
        ligand_data = fh.read()
    viewer.addModel(ligand_data, "sdf")
    viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}})

    viewer.zoomTo()
    return viewer


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    print("=" * 60)
    print("  Structure-Based Validation — Molecular Docking Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load candidates
    df = load_candidates(INPUT_CSV)

    # Step 2: Generate 3D structures (.sdf)
    records = prepare_ligand_sdfs(df, OUTPUT_DIR)
    if not records:
        sys.exit("[ERROR] No valid 3D structures generated. Exiting.")

    # Step 3: Convert ligands to .pdbqt
    records = prepare_ligand_pdbqts(records, OUTPUT_DIR)
    if not records:
        sys.exit("[ERROR] No valid PDBQT ligands. Exiting.")

    # Step 4: Prepare protein
    protein_pdbqt = prepare_protein(OUTPUT_DIR)

    # Step 5: Run docking
    records = run_docking(records, protein_pdbqt)

    # Step 6: Rank results
    results_df = rank_results(records)

    # Step 7: Save results
    save_results(results_df, OUTPUT_CSV)

    # Step 8: Visualisation hint
    # Show visualisation for the best-scoring molecule
    if len(records) > 0:
        best = min(records, key=lambda r: r.get("docking_score", 0))
        clean_pdb = os.path.join(OUTPUT_DIR, f"{PDB_ID}_clean.pdb")
        best_sdf = best.get("sdf_path", "")
        if os.path.isfile(clean_pdb) and os.path.isfile(best_sdf):
            visualise_docking(clean_pdb, best_sdf)

    print()
    print("=" * 60)
    print("  Pipeline complete ✓")
    print(f"  Results → {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
