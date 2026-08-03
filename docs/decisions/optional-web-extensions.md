# Optional web extension decisions

- Status: accepted — defer both extensions beyond Milestone 7
- Decision bead: `gpt-training-sandbox-549.7`
- Source tickets: WEB-028 and WEB-029

## Context

The required web surface is the local-only FastAPI application and plain
HTML/CSS/JavaScript client. It deliberately exposes checkpoint loading,
streaming, cancellation, session state, inspection, and transcript behavior
instead of hiding those mechanics behind a UI framework. Its deterministic
backend and real-browser acceptance paths are already covered by the Milestone
7 stack.

The two remaining roadmap items are optional extensions, not gaps in that
acceptance path:

| Extension | Milestone 7 decision | Why | Post-milestone follow-up |
| --- | --- | --- | --- |
| Gradio prototype (WEB-028) | Defer | A second UI adapter adds dependency and lifecycle maintenance without currently adding distinct educational value. | `gpt-training-sandbox-549.8` |
| Side-by-side checkpoint comparison (WEB-029) | Defer | Reproducible comparison needs a separately reviewed single-GPU execution and comparability contract. | `gpt-training-sandbox-549.9` |

The decision bead remains the sole owner of WEB-028 and WEB-029. The follow-up
beads use distinct references and depend on the Milestone 7 gate, so neither
silently becomes a Milestone 7 requirement.

## Gradio prototype

The plain client now proves the full transport and browser path directly.
Adding Gradio today would require another adapter to maintain streaming,
cancellation, reset, error handling, generation ownership, transcript export,
and privacy behavior while teaching less about those mechanics. That cost is
not justified by a demonstrated use case.

The existing `demo` dependency extra remains reserved for optional demos; no
Gradio import or entry point is added. Follow-up bead
`gpt-training-sandbox-549.8` can revisit the decision after Milestone 7. Any
future implementation must be a lazy, optional adapter over `ChatEngine` and
the shared renderer, stop set, `TokenEvent` schema, generation lease,
transcript, and privacy contracts. The plain client remains the primary UI,
and no second inference loop or session policy is permitted.

## Checkpoint comparison

Two always-resident checkpoints are not an acceptable default for the target
single RTX 3090. Follow-up bead `gpt-training-sandbox-549.9` starts from
sequential one-model residency: freeze one comparison request, run checkpoint
A, release it, load checkpoint B, and run the same request. Simultaneous
residency may only be considered after an explicit memory preflight includes
both model allocations and generation headroom.

A useful comparison must also:

- hold the prompt, seed, generation settings, context policy, and stop behavior
  constant;
- require compatible tokenizer and renderer identities, or clearly mark the
  results as non-comparable;
- cancel the active run and prevent later stages from starting, without
  committing a partial turn or leaking an engine;
- report load, time-to-first-token, decode, and total timings separately; and
- reuse `ChatEngine`, `TokenEvent`, the generation lease, transcript export,
  and privacy policy rather than adding another sampling path.

This design work is valuable, but it is independent of proving that one local
checkpoint can be operated safely through the shared engine. It therefore
belongs after the Milestone 7 gate.
