# Contributing to Viso

Thanks for your interest in contributing! This document covers the tools you
need, the conventions we follow, and how to get everything running.

## Getting started

Make sure you have the Rust toolchain and any platform-specific dependencies
installed first (see the [Prerequisites](README.md#prerequisites) section in
the README).

### Install just

[just](https://github.com/casey/just) is our task runner. All common workflows
(`check`, `fmt`, `test`, etc.) are defined in the `justfile`.

```sh
# macOS
brew install just

# Cargo (any platform)
cargo install just

# Arch Linux
pacman -S just

# Windows
winget install Casey.Just
```

### Repo setup

Run once after cloning to activate the commit-msg hook and commit template:

```sh
just setup
```

### Optional tools

These are only needed for `just check-all`:

- [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) (dependency audit):
  `cargo install cargo-deny`
- [cargo-machete](https://github.com/bnjbvr/cargo-machete) (unused dependency
  detection): `cargo install cargo-machete`

## Lint policy

Lint rules live in `Cargo.toml`: the package has `[lints] workspace = true`,
and the rules themselves are in the `[workspace.lints.clippy]`,
`[workspace.lints.rust]`, and `[workspace.lints.rustdoc]` tables. That is the
single source of truth. (`src/lib.rs` carries only `#![deny(deprecated)]`.)

Highlights:

- `clippy::all`, `clippy::pedantic`, `clippy::nursery`, and `clippy::cargo` are
  all **denied** (at priority -1, so individual lints can override). Several
  pedantic lints are explicitly allowed (e.g. the `cast_*` family, `float_cmp`,
  `similar_names`, `use_self`), so pedantic is not denied without exception.
- `missing_docs` is denied: every public item needs a doc comment.
- `clippy::unwrap_used` and `clippy::expect_used` are denied in library code.
  Tests and `main.rs` may use `#[allow]` where appropriate.
- Complexity lints are mixed: `cognitive_complexity` and `excessive_nesting`
  remain denied via the `nursery` group, with thresholds in `clippy.toml`. By
  contrast `too_many_lines` and `too_many_arguments` are explicitly **allowed**
  in `Cargo.toml`, so their `clippy.toml` thresholds are currently inert. The
  `max-fn-params-bools` / `max-struct-bools` thresholds are live
  (`fn_params_excessive_bools` / `struct_excessive_bools` stay denied).

## Running checks locally

```sh
just check        # fmt + clippy + test + doc
just check-all    # above + cargo-deny + machete + file-lengths (mirrors CI)
```

Or run the commands individually:

```sh
cargo +nightly fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

## Formatting

We use **nightly rustfmt** (config in `rustfmt.toml`). Run `just fmt` (or
`cargo +nightly fmt`) before pushing.

## Documentation

Every public struct, enum, trait, function, method, field, and variant must have
a doc comment. Use backticks for code references in doc comments (rustdoc will
warn otherwise). Functions that return `Result` should include an `# Errors`
section.

## File length limit

Source files should stay under **500 lines**. A few legacy files are above this
today (CI currently enforces 800 as a transitional limit). If you are adding to
a file that is already long, consider splitting it.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/) (lowercase,
no scope). Every commit subject must match:

```
<type>: <subject>
```

Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`.

Rules:

- **Subject line**: imperative mood, lowercase, no trailing period, max 72 chars
  (aim for 50).
- **Body** (optional): separated by a blank line, wrap at 72 chars. Explain
  *why*, not *what*.

Use `git commit` (without `-m`) to get the guided template in your editor.

### Examples

Good:

```
feat: add camera orbit controls
fix: prevent panic on empty mesh buffer
refactor: extract uniform bind group into helper
```

Bad:

```
updated stuff          # no type prefix
feat: Add Feature.     # uppercase, trailing period
fix: things            # vague subject
```

## Pull requests

- Use a descriptive title that summarizes the change.
- Link any related issues.
- Keep PRs focused on a single concern. Avoid bundling unrelated changes.
- All CI checks must pass before merge.

## Architecture

For a deeper look at the rendering pipeline, scene graph, and animation system,
see the companion [mdBook documentation](docs/).
