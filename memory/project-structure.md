---
name: project-structure
description: Document organization principles and structural decisions for the Tracel study notes project
metadata:
  type: project
---

The Tracel study notes project has document tracks:
- **System-design articles** (6): independent analysis from design decisions to source
- **Chapter tutorials** (3 completed, 16 planned): numbered walkthroughs with exercises
- **Navigation pages** (3): burn/summary.md, cubecl/summary.md, cubek/summary.md
- **Tool docs** (2): concept-index.md, SOURCE-VERSION.md

## Reading chain

`architecture → 全景 → Fusion → Autotune → JIT → CubeK → Autodiff → 全景`

Six articles form a closed-loop reading chain. Each article has prev/next navigation footer + cross-reference inline links. New articles must update neighboring articles' navigation.

## Source verification

All 84+ verifiable factual claims verified against:
- burn `78f10aec1` (2026-06-10)
- cubecl `35b861d0` (2026-06-12)
- cubek `c6a0bf40` (2026-06-12)
- burn-onnx `846b2452` (2026-06-11)

Drift tracking in `docs/SOURCE-VERSION.md` — API dependency matrix per article + high/medium/low risk ratings. Known drift: cubecl `Variable → Value` refactor (cosmetic naming change, concepts accurate).

## src/ exercises

7 workspace crates, 4 with real code + tests:
- `burn-test` — Fusion log observation
- `autodiff-test` — Autodiff gradient verification
- `ch1-gelu-variants` — GELU kernel variants
- `ch2-expand-study` — Macro expansion observation
- `ch3-trait-study`, `fusion-ch2-queue`, `fusion-ch3-drain` — skeletons

All crates compile against current reference repo commits. `cargo check --workspace` in `src/` for validation.

## Writing conventions

CLAUDE.md 8 prohibited patterns, enforced in all article types. .cursor/rules/writing-style.mdc is the Cursor copy. Concept-index.md covers ~70 key concepts with article location references.

## Restructure history

- 2026-06-01: Initial chapter structure (summary/index/N-title)
- 2026-06-09: Restructured around 5 system-design articles, added appendix/
- 2026-06-09: Added navigation footers, cross-references, one-line summaries
- 2026-06-09: Patched exercises, added autodiff-test crate
- 2026-06-10: Added CubeK system-design article (6th article)
- 2026-06-10: Added concept index, source version management
