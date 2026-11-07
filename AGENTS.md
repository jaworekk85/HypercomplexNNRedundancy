# Guidance for coding agents

Start with `docs/CURRENT.md`. Read `PROJECT.md` and `LITERATURE.md` before writing code. Treat `references.bib` as the canonical citation database.

Rules:
- Keep the scientific question broad but the first experiment small. Do not drift back into Hamilton-vs-random-algebra experiments.
- Never claim a novel method, a performance gain, or first-in-literature status without evidence.
- Confirm quaternion layers train correctly (tracked variables, gradients, updates) before any experiments.
- Distinguish real-valued trainable parameters, quaternion units, masked sparsity, and actual compact-architecture savings.
- Keep hypotheses, measured results, and interpretations in separate sections. Document versions/seeds and use held-out data.
- Always report a negative result and any failed checks accurately. Cite literature with keys found in `references.bib`.

Phase 0 and many later studies are complete. Sources and configurations registered in experiment protocols are hash-frozen: add new modules instead of editing them. Consult `docs/CURRENT.md` for live-state checks, the next action and an index of detailed records. Read full reports and archived history only when needed; scope searches to the relevant files and avoid loading all results or logs into context.
