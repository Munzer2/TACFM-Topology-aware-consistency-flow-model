# TorsionFlow

Few-step molecular conformer generation by following a **curved path** (twisting
rotatable bonds) instead of a straight line through x,y,z space.

## The question

> At 1-2 sampling steps, does the curved torsion-angle path beat the straight
> Cartesian path? And which loss function trains it best?

## What is already known

| Piece | Exists? | Where |
|---|---|---|
| Curved torsion path for conformers | yes, but slow (20 steps) | Torsional Diffusion (2022) |
| Curved rotation path for conformers | yes, 50 steps | GO-Flow (2026) |
| Few-step (1-2) conformers, straight path | yes | AvgFlow, EC-Conf, EnFlow |
| Few-step on tori, not molecules | yes | RCM (NeurIPS 2025), Generalised Flow Maps |
| Torsional flow matching, other tasks | yes | FlowPacker (side chains), MOFFlow-2 |
| **Few-step + torsion path + small-molecule conformers** | **not found** | **this folder** |

This is a combination gap: every ingredient exists, the combination does not.

Known risk: torsion angles are multi-modal (each rotatable bond prefers ~3 angles),
and a 1-step model may average modes instead of picking one. The same failure the
graph coupling probe measured in `Codes/SBM/coupling_probe.py`.

**QM9 floor.** A torsion-only model keeps RDKit's bond lengths and angles, which caps
GEOM-QM9 accuracy near AMR-R 0.17 A. Torsional Diffusion already reaches 0.178. So
on QM9, compare few-step torsion models against Torsional Diffusion (same floor),
and leave torsion-vs-Cartesian comparisons to GEOM-Drugs (step 8). Details in
`step2_reproduce/README.md`.

## Start here

**The pipeline:** `notebooks/pipeline_qm9.ipynb` trains our twist-angle flow model
on real GEOM-QM9, generates conformers for the official test molecules in 1, 2, 5
and 20 steps, and scores them with the benchmark's own script. It reuses Torsional
Diffusion's data loading, network and scoring (patched), with our flow-matching
training and few-step sampler. `MODE = "smoke"` checks every piece on a CPU in
~3 minutes; `MODE = "full"` is the real run and needs the GPU. Built in checks: the
path lands back to ~1e-7 Å, the lowest possible loss (0.48 vs 3.29 for "predict 0"),
a memorisation test proving the network can learn, and validation error by time.
`score_checkpoint.py` scores the current `best_model.pt` on the first 100 test molecules
(next to RDKit on the same molecules) from a second terminal while training continues.

**The explainer:** `notebooks/training_walkthrough.ipynb` walks through the whole training procedure,
one cell at a time: data, rotatable bonds, conformer matching, the curved path,
the model, one training step by hand, three loss functions, the training loop, and
generation scored like the real benchmark. Toy molecules and a small model, so it
runs on a CPU in about 8 minutes (one thread, exactly repeatable). Needs only
torch, rdkit, numpy, matplotlib. Unfamiliar terms: see `GLOSSARY.md`.

Part J of the notebook tests research questions 1-2 (fake vs real multiple answers):

- **Q1 (exact, no model):** on the toy set, 42% of the ambiguity at the start of
  generation is "fake" (symmetric groups whose positions are the same molecule);
  81% of that comes from CH3 groups, which only move hydrogens.
- **Q2 (5 seeds, exactly repeatable runs):** treating symmetric twists as n x angle
  improves molecules whose symmetric group moves heavy atoms, most consistently at
  1-2 steps (about 0.025-0.035 A AMR-R, ~8-12%, 4-5 of 5 seeds). It is not free:
  fixing CH3 groups too makes plain chains worse at 1-2 steps (+0.025 A at 1 step,
  every seed). Fixing heavy groups only keeps the gain and halves the cost; the
  rest of the cost comes indirectly, through the one network all molecules share.
- **Best version so far:** fix heavy-atom symmetric groups only.
- **Next:** measure Q1 on real GEOM conformers (no training, CPU); then understand
  the shared-network cost.

## Steps

Each step has a pass condition. Do not start the next step until it passes.

- [ ] **1. Environment** (`step1_setup/`) - native Windows, CUDA 12.8+ PyTorch on
      the RTX 5080. No WSL.
      *Pass:* `check_env.py` reports all PASS.
- [ ] **2. Reproduce** (`step2_reproduce/`) - patch Torsional Diffusion for Windows
      and current libraries (`patch_windows.py`, 9 patches, verified), then
      reproduce paper Table 7 on GEOM-QM9 in three tiers: RDKit baseline (no
      model), the authors' checkpoint, our own training. Time one training run so
      later steps can be sized.
      *Status:* data downloaded and unpacked; all their scripts (train, generate,
      evaluate) run on Windows. Patch 7 matters most: unpatched on Windows, 784 of
      the 1000 test molecules would land in training. Remaining: the real runs (GPU).
      *Pass:* AMR within 0.01 A and COV within 1.5 points of Table 7.
- [x] **3. Probe** (`step3_probe/`) - no training. Research question 1 on all of
      GEOM-QM9 (108,563 molecules): 30% of the ambiguity a twist-angle model faces
      at t=0 is fake, but 98.6% of that comes from H-only groups (CH3, OH...), which
      the score cannot see. Of the ambiguity the score CAN see, only 0.8% is fake:
      heavy-atom symmetric groups are rare in QM9 (2.3% of molecules). Within those
      molecules it is 38%. Boltzmann weighting gives the same answer. **So the
      symmetry fix cannot move QM9 benchmark numbers much; whether it matters
      depends on drug-like molecules (GEOM-Drugs), where phenyl rings and CF3
      groups are common.** Results: `step3_probe/qm9_results.txt`.
- [ ] **4. Torsional flow matching** (`notebooks/pipeline_qm9.ipynb`) - swap
      diffusion for flow matching on the torus. Multi-step model first.
      *Status:* pipeline built and runs end to end (smoke mode, CPU). Verified: the
      path lands back within ~1e-7 Å; the network memorises 16 fixed examples
      (3.7 -> 0.02); the sampler and benchmark scoring work. NOT yet verified:
      learning the real task. On CPU (up to 6000 steps) the loss stays at
      "predict 0" (3.29) although 0.48 is possible; Torsional Diffusion's own
      training behaves the same early on. Needs the full run on the GPU.
      *First GPU result (2026-10-15, mid-training checkpoint, epoch 80+ of 250,
      `score_checkpoint.py`, first 100 test molecules):* it learns. AMR-R 0.164 at 20
      steps vs RDKit 0.237 on the same molecules (paper, all 1000: RDKit 0.235, TD 0.178).
      Step curve, AMR-R / AMR-P: 1 step 0.200 / 0.256, 2 steps 0.190 / 0.248, 5 steps
      0.170 / 0.216, 20 steps 0.164 / 0.201 (RDKit 0.237 / 0.204). 6 of 100 molecules
      fail RDKit embedding (same in every row, counted as 0 coverage).
      *Final (2026-10-16, epoch 250, all 1000 test molecules):* COV-R / AMR-R / COV-P / AMR-P:
      random twists 82.71 / 0.248 / 73.31 / 0.323; 1 step 83.78 / 0.236 / 77.34 / 0.296;
      2 steps 85.93 / 0.208 / 80.29 / 0.266; 5 steps 86.86 / 0.188 / 81.95 / 0.240;
      20 steps 86.83 / 0.182 / 82.45 / 0.230 (paper TD 92.8 / 0.178 / 92.7 / 0.221).
      Validation loss flat from ~epoch 100 (2.31 at the end). AMR-R is within 0.004 A of
      TD. COV is not comparable yet: RDKit 2026.09 fails to embed 81 test molecules
      (strained cages with several stereocentres; `enforceChirality=False` rescues 76), and
      those count as 0 coverage. TD's 92.8 COV-R is impossible with that many failures, so
      the paper's RDKit must have embedded more of them. Fair comparison: run TD's own
      checkpoint through this same pipeline.
      *Pass:* the 20-step model comes close to Torsional Diffusion's AMR-R 0.178.
      If the loss hasn't moved after ~10 full epochs, give the network the twist
      angles as explicit inputs.
- [ ] **5. Loss screening** - velocity MSE, endpoint prediction, wrapped-angle
      loss, pseudo-Huber, time weighting, 3D-space loss, distribution loss.
      1 seed, short runs, keep the top 2-3.
- [ ] **6. Few-step** - Reflow, consistency (RCM-style, wrapped distance),
      distillation, using the best loss.
- [ ] **7. Compare** - against AvgFlow-distill and EC-Conf with ONE shared
      evaluation script, plus independent physical checks (force-field energy,
      clashes, bond geometry) so we are not graded only by an RMSD-based metric.
- [ ] **8. Scale** - if promising, repeat on GEOM-Drugs (bigger, floppier
      molecules, where torsion methods should matter more).

## Base code

- `gcorso/torsional-diffusion` (PyTorch) - conformer matching, torsion network,
  GEOM-QM9 support (`--dataset qm9`), evaluation scripts.
- Baselines: `NVIDIA-Digital-Bio/avgflow` (JAX), EC-Conf (released code).

## Sources

- Torsional Diffusion - https://arxiv.org/abs/2206.01729
- RCM - https://arxiv.org/abs/2510.00983 (also `Resources/12236_Riemannian_Consistency_M.pdf`)
- Flows, straight but not so fast (Riemannian Reflow) - https://arxiv.org/abs/2510.24732
- Generalised Flow Maps - https://arxiv.org/abs/2510.21608
- AvgFlow - https://arxiv.org/abs/2507.09785
- EC-Conf - https://arxiv.org/abs/2308.00237
- GO-Flow - https://arxiv.org/abs/2605.25577
- FlowPacker - https://www.biorxiv.org/content/10.1101/2024.07.05.602280
