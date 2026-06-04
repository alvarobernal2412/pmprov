# Contributing to pmprov

Thank you for your interest in contributing to `pmprov`! This is a PhD research prototype, so contributions are especially welcome in the form of bug reports, reproducibility feedback, and targeted improvements aligned with the research goals.

---

## Ways to contribute

- **Bug reports** — unexpected behaviour, crashes, or silent failures in the middleware
- **Reproducibility issues** — anything that makes provenance records incomplete or incorrect
- **Test coverage** — the test suite is sparse; new tests for `tracker/` and `models/` are high-value
- **Documentation** — clarifications, corrections, or examples
- **Feature proposals** — open an issue first so we can discuss alignment with the research scope

---

## Development setup

**Prerequisites:** Python ≥ 3.13, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/alvarobernal2412/pmprov.git
cd pmprov
uv sync --all-extras --dev
```

Run the tests:

```bash
uv run pytest
```

---

## Workflow

1. **Open an issue** before starting non-trivial work — this avoids duplicated effort and lets us discuss scope.
2. **Fork** the repository and create a branch from `main`.
3. **Write tests** for any change to `tracker/` or `models/`. Bug fixes should include a regression test.
4. **Keep commits focused** — one logical change per commit, conventional commit messages preferred (`feat:`, `fix:`, `test:`, `chore:`, `docs:`).
5. **Open a pull request** against `main`. Fill in the PR template.
6. **Respond to review** — the maintainer aims to review within a week.

---

## Design constraints

- The provenance structure is a **tree**, not a DAG — branch merging is intentionally unimplemented until research motivates it.
- All Pydantic models use **FK-style string IDs** — no object references between models.
- `trace_step` must **never break analyst code** — errors are caught and silenced.
- All timestamps are **UTC-aware** (`datetime.now(timezone.utc)`).

---

## Reporting bugs

Use the [Bug Report](https://github.com/alvarobernal2412/pmprov/.github/ISSUE_TEMPLATE/bug_report.md) template. Please include:

- A minimal reproducible example (ideally a notebook cell sequence)
- The function call that was being tracked when the issue occurred
- Whether the issue is in provenance capture, storage, or visualisation

---

## Code of conduct

Be respectful and constructive. This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1.

---

## Questions

Open a [Discussion](https://github.com/alvarobernal2412/pmprov/discussions) or reach out via the contact on the author's GitHub profile.
