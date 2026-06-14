---
name: project-structure
description: Document organization principles and structural decisions for the Tracel study notes project
metadata:
  type: project
---

The Tracel study notes project has dual document tracks: system-design articles (独立成篇的分析) and chapter tutorials (编号跟练教程). Decks used for documentation are Markdown; exercises are Rust crates in `src/`.

## Core structure (2026-06-09 restructure):

1. **Five system-design articles** form the primary content, covering Burn/CubeCL's four core systems:
   - `docs/burn/burn-systems-architecture.md` — full-stack panorama (recommended entry point)
   - `docs/burn/kernel-fusion-system-design.md` — lazy queue fusion engine
   - `docs/cubecl/autotune-system-design.md` — strategy enumeration autotuner
   - `docs/cubecl/jit-compilation-pipeline.md` — #[cube] → IR → GPU binary pipeline
   - `docs/burn/autodiff-system-design.md` — decorator pattern autodiff

2. **Chapter tutorials** remain as numbered docs (`N-title.md`) with corresponding `src/` crates. Completed: CubeCL ch1 (gelu), CubeCL ch2 (expand), Fusion ch1 (client-server).

3. **Navigation pages** (`summary.md` in burn/ and cubecl/) redirect to both system-design articles and chapter tutorials.

4. **Appendix** (`docs/appendix/`) holds archived/translated content.

## Key structural decisions:

5. **Old blog translation** (`automatic-kernel-fusion.md`) moved to appendix — superseded by system-design articles.

6. **Cross-project architecture** (`docs/architecture.md`) updated with links to all five system-design articles.

7. **Writing conventions** (CLAUDE.md 8 prohibited patterns) unchanged — they apply to all document types.

8. **src/** structure unchanged — existing exercise crates continue to complement chapter tutorials.

**Why:** The project evolved from chapter-style source walkthroughs to system-design analysis. System-design articles provide the "why" and comparative context that backend/Infra engineers need; chapter tutorials remain valuable for hands-on mechanism tracing.

**How to apply:** New system-design articles go into the appropriate `docs/<project>/` directory with `-system-design.md` suffix. New chapter tutorials follow the existing `N-title.md` + `src/<crate>/` pattern.
