# Guidance for coding agents

Read `PROJECT.md` and `LITERATURE.md` before writing code. Treat `references.bib` as the canonical citation database.

Rules:
- Keep the scientific question broad but the first experiment small. Do not drift back into Hamilton-vs-random-algebra experiments.
- Never claim a novel method, a performance gain, or first-in-literature status without evidence.
- Confirm quaternion layers train correctly (tracked variables, gradients, updates) before any experiments.
- Distinguish real-valued trainable parameters, quaternion units, masked sparsity, and actual compact-architecture savings.
- Keep hypotheses, measured results, and interpretations in separate sections. Document versions/seeds and use held-out data.
- Always report a negative result and any failed checks accurately. Cite literature with keys found in `references.bib`.

Phase 0 and many later studies are complete. Current state, conventions (hash-frozen sources:
add new modules instead of editing them) and how to resume: `docs/HANDOFF.md`. Priorities and
paper story: `docs/PLAN_IJCNN2027.md`. Measured results: `docs/RESEARCH_STATUS.md`.
