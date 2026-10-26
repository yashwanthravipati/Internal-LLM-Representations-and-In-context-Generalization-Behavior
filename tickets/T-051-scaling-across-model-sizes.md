---
id: T-051
title: (Stretch) Scaling across model sizes
status: todo
phase: 5
priority: P2
depends_on: [T-024, T-033]
owner: unassigned
estimate: M
tags: [stretch, scaling]
updated: 2026-10-25
---

# T-051 — (Stretch) Scaling across model sizes

## Context
Does the link between geometry and behavior get stronger (or weaker) with scale? Do larger models shift their representations more? Pre-fitted lenses exist for Qwen3.5-0.8B, Qwen3.5-2B, Qwen3-4B and Qwen3.5-9B. **Start only after M2.**

## Tasks
- [ ] Re-run the core analyses (Exp 1 primary metric, the H2 key figure, H3, and H4/H5 if feasible) on every available size the compute allows.
- [ ] Plot each primary metric vs model size, with layers on normalized depth.
- [ ] Note the confound: Qwen3 vs Qwen3.5 differ in family as well as size, so compare within one family where possible.

## Acceptance criteria
- [ ] Scaling figure plus a short write-up in `results/scaling/REPORT.md`.

## Outputs
- `results/scaling/REPORT.md`, `figures/scaling_*`.

## Notes & pitfalls
- 9B in bf16 likely exceeds a 16 GB GPU. Use 8-bit loading (the course list allows this for larger models), and check that activations and lens readouts aren't distorted by quantization.

## References
- PLAN §4.1, §4.5

## Work log
