# Glossary: the terms used in this project, in plain language

Start with "The story in 8 sentences". Then look up any term below.

---

## The story in 8 sentences

1. We want a program that, given a molecule's drawing (which atoms, which bonds),
   produces realistic **3D shapes** of it.
2. Most of a molecule's shape comes from **twisting** around its single bonds, so we
   keep bond lengths and angles fixed and only generate the **twist angles**.
3. The model learns by watching shapes **move from random to real**, and learning the
   speed and direction of that movement (**flow matching**).
4. Moving by twisting bonds follows a **curved path** through 3D space, which keeps
   every bond intact; a **straight path** would break bonds on the way.
5. To generate, start from a random shape and **follow the learned movement** for a
   number of **steps**; fewer steps = faster.
6. With **one step**, quality drops, because the model only learns the **average**
   direction toward several possible real shapes, and one jump lands between them.
7. Some of those "several possible shapes" are **fake**: a CH3 group turned by 120°
   is the identical molecule, but a 360° circle treats it as a different answer.
8. Our idea: **remove the fake ones exactly** (measure such angles modulo 120°), and
   test whether that makes 1-step generation better.

---

## Molecules and chemistry

**Conformer.** One 3D shape of a molecule. The same molecule can have many.
Butane, for example, can be stretched out ("anti") or bent ("gauche").

**Heavy atom.** Any atom that isn't hydrogen (C, N, O, F). The benchmark score only
compares heavy atoms, because hydrogen positions are uncertain anyway.

**Bond length / bond angle.** The distance between two bonded atoms, and the angle
between two bonds on one atom. Both barely change between conformers.

**Torsion angle (twist angle, dihedral).** The twist around a bond, measured with 4
atoms in a row (a-u-v-b): how far atom b is turned around the u-v bond relative to
atom a. This is what changes between conformers. "Torsion", "twist" and "dihedral"
mean the same thing here.

**Rotatable bond.** A bond you can twist around. Our rule (same as Torsional
Diffusion): cutting it splits the molecule in two, and the smaller piece has more
than one atom. Bonds inside rings don't count.

**Local structure.** The fixed part of the geometry (all bond lengths and angles),
taken from one RDKit shape. A torsion model only ever changes the twists.

**Conformer matching.** Taking a real conformer and rewriting it as "local structure +
twist angles": measure its twists, apply them to the fixed local structure.

**Floor.** The best accuracy any twist-only model can reach, because the local
structure isn't exactly right. On GEOM-QM9 it's about 0.17 Å.

**Methyl group (CH3), CF3, tert-butyl, phenyl.** Common groups of atoms. CH3 is a
carbon with three hydrogens; CF3 the same with fluorines; tert-butyl is a carbon with
three CH3 groups; phenyl is a 6-carbon ring.

**Symmetric group / symmetry number.** A group that looks identical after a partial
turn. CH3, CF3 and tert-butyl repeat every 120° (symmetry number 3); a phenyl ring
every 180° (symmetry number 2).

**RDKit.** Free chemistry software. It reads molecules, makes rough 3D shapes (with a
method called **ETKDG**), and measures shapes.

**Force field (MMFF).** A cheap physics model of a molecule's energy. "Relaxing" a
shape with it moves the shape into the nearest low-energy position.

**GEOM-QM9 / GEOM-Drugs.** The standard datasets. QM9: ~133k small molecules (at most
9 heavy atoms). Drugs: ~300k bigger, drug-like molecules. Each comes with many real
conformers, computed with an accurate method called **CREST**.

---

## Measuring quality

**Å (ångström).** 0.1 nanometres. A C-C bond is about 1.5 Å long.

**RMSD.** Root-mean-square distance between matching atoms of two shapes, after
rotating one to line up with the other as well as possible. 0 = identical.

**Kabsch alignment.** The standard way to find that best rotation.

**AMR-R (recall).** For each *real* conformer, find the closest *generated* one and
take the RMSD; average those. "Did we cover the real shapes?" Lower is better.

**AMR-P (precision).** The other way round: for each *generated* conformer, the RMSD
to the closest real one. "Is each generated shape good?" Lower is better.

**COV-R / COV-P (coverage).** The percentage of real (or generated) conformers with a
partner within 0.5 Å. Higher is better. Noisy when there are few conformers.

---

## How the model learns and generates

**Generative model.** A model that produces new examples (here: 3D shapes) instead of
predicting a label.

**Prior / random start / noise.** Where generation begins. For us: the local structure
with every twist angle chosen at random.

**Path (interpolation).** The route from a random start to a real conformer that we
show the model during training.

**Straight path vs curved path.** Straight = every atom moves in a straight line in
x,y,z (can squash bonds). Curved = we change twist angles, so atoms swing around bonds
on arcs (bonds stay intact).

**Geodesic.** The shortest path on a curved surface. On a circle of angles, it means
turning the short way round.

**Wrap-around.** Angles repeat every 360°. Going from 170° to -170° is a 20° turn, not
340°. `wrap()` in the code handles this.

**Manifold / torus.** A space with its own shape. A torus is a doughnut: the space of
several angles, each wrapping around. Our twist angles live on a torus.

**Velocity (velocity field).** How fast and in which direction each twist angle should
change at a given moment. This is what the network predicts.

**Flow matching.** The training method: show the model points along many paths
(random start -> real conformer), and train it to predict the velocity at each point.

**Training target.** The correct answer the model is compared to. For us: the true
velocity along the path.

**Loss / MSE.** A number saying how wrong the model is. MSE (mean squared error) =
average of (prediction - target)².

**Training step / batch / epoch.** One step = one update of the model, using one
batch (here, 64 examples). One epoch = one pass through the whole dataset.

**Sampling steps (inference steps).** How many small moves generation takes from the
random start to the final shape. 20 steps = 20 calls to the model. 1 step = one jump.

**Few-step / 1-step generation.** Generating in 1-2 steps instead of 20+. Faster, but
usually worse. Making it as good is the project's goal ("gap 2").

---

## Why 1-step generation fails, and the methods that try to fix it

**Mode / multimodal.** A mode is one of several typical answers. A twist angle that is
usually 60°, -60° or 180° has three modes: it is multimodal.

**Mode averaging.** When several answers are possible, an MSE-trained model learns
their average. One jump along that average lands between the real shapes.

**Ambiguity.** Our word for "several destinations are possible from here". We measure
it as the spread of the moves needed to reach them.

**Fake vs real multiple answers.** *Real*: genuinely different shapes (anti vs
gauche). *Fake*: positions of a symmetric group that are the same molecule (a CH3
turned by 120°). Fake ones can be removed exactly; real ones can't.

**Symmetry fix (n x angle).** For a group that repeats every 360°/n, the model works
with n times the angle. Then identical positions become the same point.
"Heavy-only fix" = apply this only to groups that move heavy atoms (CF3, tert-butyl,
phenyl) and leave CH3 at 360°.

**Coupling.** Which random start gets paired with which real conformer during
training. Pairing similar ones can make learning easier.

**Reflow (rectified flow).** Train a model, use it to produce (start -> output) pairs,
then retrain on those pairs. The paths get straighter, so fewer steps are needed.

**Distillation (teacher / student).** A slow, accurate model (teacher) trains a fast
model (student) to copy its results in fewer steps.

**Consistency model.** A model trained so that from *any* point along a path it jumps
straight to the same end point. Made for 1-step generation.

**Symmetry / invariance / equivariance.** Invariant: the output doesn't change when
the input is rotated (e.g. a twist angle). Equivariant: the output rotates along with
the input (e.g. a predicted 3D direction).

**SO(3).** The set of all 3D rotations. "Handling SO(3)" means dealing with the fact
that a rotated molecule is still the same molecule.

---

## Papers mentioned

| Name | One line |
|---|---|
| **Torsional Diffusion** (2022) | Generates twist angles on a torus; 20 steps; our base code. Treats every twist as 360°. |
| **AvgFlow** (2025) | Straight paths in x,y,z; averages over whole-molecule rotations; ~8x faster training; 1-2 steps via Reflow + distillation. |
| **GO-Flow** (2026) | Splits motion into move + rotate (curved) + shape; 50 steps. |
| **RCM** (2025) | Consistency models on curved spaces (tori, spheres, rotations); 2-step generation. |
| **"Flows, straight but not so fast"** (2025) | Reflow on curved spaces for proteins; helps only a little, mostly at low step counts. |
| **FlowPacker** | Flow matching on twist angles, for protein side chains. |
| **EC-Conf** | Consistency model for conformers in x,y,z; 1 step. |
| **SymDrift** (2026) | 1-step generation that removes whole-molecule rotation ambiguity. |
| **SubFlow** | Image generation; fixes mode averaging by telling the model which mode to aim for. |
| **CDLM** (2026) | Consistency training for text; averages over paths. |

---

## Experiment words

**Seed.** The starting value of the random number generator. Same seed = same random
choices. Running several seeds checks that a result isn't luck.

**Run-to-run noise / determinism.** With several CPU threads, numbers get added in a
different order each run, so results drift slightly. With one thread, runs are
exactly repeatable (deterministic).

**Baseline / arm.** Baseline: the version without the new idea. Arm: one version in a
comparison (here: "360°", "fix all groups", "fix heavy groups only").

**Paired comparison.** All versions start from exactly the same random shapes, so
differences come from the models, not from luck in the starting points.

**Ablation / controlled experiment.** Change one thing, keep everything else the same,
see what that one thing does.

**[k/5] and p.** "[4/5]" = the change went the right way in 4 of 5 seeds. p = how
often a split at least that lopsided would happen by pure chance if the idea did
nothing. Small p = less likely to be luck. With only 5 seeds, the smallest possible
two-sided p is 0.06.
