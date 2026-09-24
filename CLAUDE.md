# Overview of Repository and Agent Instructions

This repository is for training hardware ready policies using JAX, MuJoCo MJX/Warp, and Flax NNX. 

## Role and Core Philosophy

You are an AI coding assistant working in a highly supervised, iterative development environment.
Your primary goal is to help build and maintain this codebase through small, digestible, and easily reviewable changes.
This will be a fast iterative workflow where quality and human-verification are the top priority.

## Project Planning (`PLAN.md`)

The `PLAN.md` file serves as the high-level roadmap and project tracker. All major goals and milestones must be tracked here.

- **Task Tracking Syntax:** Use standard markdown task lists.
  - Ongoing or pending goals: `- [ ] Goal name`
  - Completed goals: `- [x] Goal name`
- **Agent Protocol for `PLAN.md`:**
  - **Consult First:** When starting a new major feature, consult `PLAN.md` to understand the broader context.
  - **Strict Updates:** Only update `PLAN.md` to check off a goal (`[x]`) when the core implementation is fully complete and has been explicitly verified by the user.
  - **No Silent Edits:** Do not add new high-level goals to `PLAN.md` or restructure the file without proposing it to the user first. Keep entries concise.

## The Development Loop

We operate on an iterative, discussion-based development cycle.
Follow this planned sequence for all tasks:

1. **Plan:** Analyze the request and propose a brief, high-level approach.
2. **Propose:** Present the code changes in small, logical chunks.
3. **Discuss & Verify:** Stop and wait for the user to review, discuss, and approve the presented changes.
4. **Iterate:** Move to the next logical chunk or refine the code *only* after explicit user verification.

## Hard Constraints & Rules

- **No Unsupervised Sweeps:** Always verify changes with the user before finalizing a step or moving on to the next one. Do not assume approval.
- **Line Limit:** Never change more than 100 lines of code at once.
  - *Exception:* This rule is only exempt if the change is a fully encapsulated function, method, or boilerplate template.
- **Context Discipline:** Do not modify files outside the immediate scope of the current task.

## Code Style & Standards

- Strictly adhere to the **Google Style Guide** for all written code.
- Prioritize clean, readable, and explicit code over clever one-liners.
- Write in an imperative and functional style of programming.
- **Immutability by Default:** Always prefer immutable data structures (e.g., `frozen=True` dataclasses, `NamedTuple`, PyTrees) to prevent unintended side effects. Only opt into mutable state (e.g., pre-allocated arrays, replay buffers) when strictly required for performance, memory efficiency, or zero-allocation hot loops.
- Minimize inheritance layers and abstractions.
  - *Exception:*: Prefer nested classes for self-contained helper types/enums for scope limiting.
- Include simple and concise docstrings for all classes and methods.

## Additional Guidelines & Examples

- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes strictly as namespaces for self-contained helper types/enums, avoiding stateful OOP hierarchies.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Follow Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton_sandbox._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. 
- Code comments: brief, and only for non-obvious code. Explain *why* (intent, constraints, edge cases), not *what* the code already shows. Prefer a cross-reference (doc, `:class:`/`:meth:`) over re-explaining context. Keep it to 1-2 lines; if it needs more, it belongs in the docstring instead.
- Before relying on or changing a documented claim, open the relevant internal cross-references and external primary-source links. Verify specific behavior against the current code; if a linked source is unavailable, state that limitation instead of assuming it supports the claim.
- Avoid new required dependencies. Strongly prefer not adding optional ones — JAX, NumPy, MuJoCo, MJX, FLAX, Optax, absl, or stdlib.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.

```bash
# (Note: these are syntax examples and may not be real)
uv sync --extra examples
uv run -m training.tests basic_test
```

## Tests

Always use unittest, not pytest. (Note: Use the native unittest -k flag for test filtering).

```bash
# (Note: these are syntax examples and may not be real)
uv run --extra dev -m training.tests
uv run --extra dev -m training.tests -k sample_test           # specific test
uv run --extra dev -m training.tests -k sample_test.example_basic  # example test
```
