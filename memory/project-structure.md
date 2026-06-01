---
name: project-structure
description: Document organization principles and structural decisions for the Tracel study notes project
metadata:
  type: project
---

The Tracel study notes project follows a three-layer document hierarchy: map (summary.md) → plan (index.md) → chapter (N-title.md). Each of the four Tracel projects (Burn, CubeCL, CubeK, Burn-ONNX) has a map document. Chapter series currently exist for Burn Fusion (8 chapters) and CubeCL (8 chapters), with completed chapters at 3/16.

## Key structural decisions (2026-06-01):

1. **Cross-project architecture narrative** at `docs/architecture.md` ties together the decision-deferral theme (L1 compile-time → L2 JIT-time → L3 first-execution) across all four projects.

2. **CubeK and Autodiff got map documents** (`docs/cubek/summary.md`, `docs/burn/autodiff/summary.md`) to fill the two biggest coverage gaps. CubeK focuses on Blueprint-Routine-Autotuner three-layer discipline; Autodiff on selective float-tensor wrapping and gradient graph.

3. **ONNX got a chapter plan** (`docs/burn/onnx/index.md`) with 6 chapters covering the IR pipeline from protobuf parsing to codegen and testing.

4. **Decision-timing framing** was added to the end of every completed chapter (fusion ch1, CubeCL ch1, CubeCL ch2) as a structural pattern to maintain.

5. **Exercise skeletons** for the next 3 chapters (CubeCL ch3, Fusion ch2, Fusion ch3) were created in `src/`. Future chapters should follow the same pattern: a Cargo crate with a stub lib.rs.

**Why:** The project had high planning-to-content ratio (16 chapters planned, 3 written), uneven coverage (no CubeK or Autodiff documents), and lacked a cross-cutting narrative connecting the four projects.

**How to apply:** When adding new content, follow the existing patterns: chapters end with decision-timing table + common-misconceptions table + homework + next-chapter preview. New projects get a map document first, then a chapter plan, then chapters.
