# ICESEE LETKF revision 3 transfer

This temporary branch transfers the implementation to Brian's Mac or PACE. KYANJO/ICESEE has a different history from ICESEE-project/ICESEE, so the code is carried in a Git bundle rather than applied to this repository's older source tree.

`ICESEE_LETKF_R3.bundle` contains the three exact implementation commits, including the complete revision 3 blueprint, source changes, test evidence, and implementation handoff.

- Implementation ref: `refs/heads/implementation/letkf-r3-20261111`
- Implementation head: `4382241f250064cbb12918f2f994277e5f622fd4`
- Required base: `e1e4ce8fd4f33a642f41c18fcec16f4374f0c1b0` from ICESEE-project/ICESEE
- Implementation tree: `886ad9befcb824b8ccdd3404cdd072582f52b8c4`
- Bundle SHA-256: `fc38c610c0fe8d62818b466967cd2984e0d61dcaea4e4d076d8d01e33631265e`

## Import on Mac or PACE

Run from an existing ICESEE clone. These commands create a separate continuation branch and worktree; they assume those names and paths are unused. If the patch or bundle was already applied, inspect the existing worktree and compare its tree before importing again. Do not reset existing work.

```bash
git fetch --no-tags https://github.com/ICESEE-project/ICESEE.git \
  e1e4ce8fd4f33a642f41c18fcec16f4374f0c1b0
git fetch --no-tags https://github.com/KYANJO/ICESEE.git \
  handoff/letkf-r3-20261111
icesee_bundle_path=$(mktemp "${TMPDIR:-/tmp}/icesee-letkf-r3.XXXXXX")
git show FETCH_HEAD:handoff/letkf-r3-20261111/ICESEE_LETKF_R3.bundle > "$icesee_bundle_path"
git bundle verify "$icesee_bundle_path"
git fetch "$icesee_bundle_path" \
  refs/heads/implementation/letkf-r3-20261111:refs/heads/continuation/letkf-r3
git worktree add ../ICESEE-letkf-r3 continuation/letkf-r3
git -C ../ICESEE-letkf-r3 rev-parse HEAD
git -C ../ICESEE-letkf-r3 rev-parse 'HEAD^{tree}'
```

The last two commands must print the implementation head and tree above. The bundle preserves the original commits without patch reapplication.

## Continue the work

In the imported worktree, read:

1. Destination `AGENTS.md` instructions, if present.
2. `docs/letkf-r3-handoff.md` for the implementation, test reproduction, environment details, and remaining gates.
3. `docs/blueprints/ICESEE_LETKF_Implementation_Blueprint_2026-11-11.pdf` for the complete LETKF, native ISSM forecast, storage, and recovery plan.

Use this transfer README for import instructions; the older transport discussion in the bundled implementation handoff predates this KYANJO transfer branch.

The focused suite recorded 60 passing tests. Four- and eight-rank MPI checks passed, including sharded checkpoint restoration. A broader test selection reproduced the same 11 failures on unchanged main; the logs are included.

Real Icepack/ISSM integration, the persistent C++ ISSM worker, MATLAB-free restart, sparse concurrent LETKF routing, and continental memory/storage qualification remain unfinished. Reproduce checks with the actual destination Python and MPI environment before simulations. Use a compute allocation for MPI on PACE and avoid existing simulation output paths.

## Retention

After import, verify the destination's continuation branch and implementation head before deleting this temporary remote transfer branch. The destination's local branch retains the imported implementation commits.
