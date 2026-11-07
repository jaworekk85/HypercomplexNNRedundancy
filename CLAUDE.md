# Research coding context

**Current handoff, 2026-11-07:** read the first section of `docs/HANDOFF.md`
before continuing. Both ResNet20 baselines and the 12-case one-seed compression
study are COMPLETE. No GPU job is running. User authorizes proceeding with the
planned focused multi-seed replication, then bounded hybrid/PHM study; conserve
their remaining weekly token budget and keep updates concise. Earlier "start
with a tiny model" wording below is historical; Phase 0 is already complete.
The user is about to switch from Codex to Claude Code. Do not ask them to repeat
project context. No messages have been sent to Marcos; draft is
`docs/MARCOS_DRAFT_2026-11-07.md` with author placeholder.

Start by reading `PROJECT.md` (requirements and experiment plan), then `LITERATURE.md` (annotated prior art) and `references.bib` (citation keys).

Focus: activation redundancy in quaternion-valued neural networks and its use for **structured** compression. Start with a tiny verified trainable model and activation analysis, not a new algorithm. Preserve four-component quaternion grouping; use fair real-valued baselines. Do not assert novelty or performance without experiments. See `AGENTS.md` for guardrails. Current state and how to resume: `docs/HANDOFF.md`; plan: `docs/PLAN_IJCNN2027.md`.
