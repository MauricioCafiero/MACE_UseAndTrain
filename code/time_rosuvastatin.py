import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import mace_calc as mc

SMI = "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C"

t0=time.time()
atoms = mc.smiles_to_atoms(SMI, n_conformers=5, seed=1)
tbuild=time.time()-t0
print(f"build (5-conf MMFF lowest): {tbuild:.1f}s  atoms={len(atoms)}  formula={atoms.get_chemical_formula()}", flush=True)
print("elements:", sorted(set(atoms.get_chemical_symbols())), "auto ->", mc.select_model(atoms), flush=True)
mc.smiles_to_xyz(SMI, "rosuvastatin.xyz", n_conformers=5, seed=1)
print("wrote rosuvastatin.xyz", flush=True)

print("\n== MACE optimize (off-medium, float64, LBFGS, fmax=0.01, max 500) ==", flush=True)
t0=time.time()
opt = mc.optimize(atoms, model="off-medium", fmax=0.01, steps=500, dtype="float64",
                   optimizer="LBFGS", logfile=None)
topt=time.time()-t0
print(f"optimize: {topt:.1f}s  converged={opt['converged']}  E={opt['energy']:.4f} eV  final fmax={opt['fmax']:.4f} eV/A", flush=True)

print("\n== MACE vibrations (off-medium, float64) ==", flush=True)
t0=time.time()
vib = mc.vibrations(atoms, model="off-medium", dtype="float64", name="rosu_vib")
tvib=time.time()-t0
freqs = vib["frequencies_cm1"]
real = np.real(freqs[np.abs(freqs.imag) < 1e-6])
imag = np.abs(freqs[np.abs(freqs.imag) >= 1e-6].imag)
print(f"vibrations: {tvib:.1f}s  nmodes={vib['nmodes']}  n_real={len(real)}  n_imag={len(imag)}", flush=True)
if len(imag): print(f"  imaginary |cm-1| (lowest 8): {np.round(np.sort(imag)[:8],1)}", flush=True)
if len(real): print(f"  lowest 6 real: {np.round(np.sort(real)[:6],1)}", flush=True)
print(f"\nTOTAL opt+freq wall: {topt+tvib:.1f}s ({(topt+tvib)/60:.2f} min)", flush=True)
