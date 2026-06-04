## What does this PR do?

<!-- One paragraph summary. Link the related issue: "Closes #123" -->

## Type of change

<!-- Check all that apply -->

- [ ] 🐞 Bug fix
- [ ] 🕵🏻 Fix (non-breaking improvement)
- [ ] 🚀 New feature
- [ ] ⚠️ Breaking change
- [ ] 🧪 Test coverage
- [ ] 📝 Documentation
- [ ] ⚙️ Configuration / CI

## Changes

<!-- Bullet list of what changed and why. Focus on decisions, not diffs. -->

-
-

## Testing

<!-- How was this tested? Include cell sequences, pytest commands, or notebook steps. -->

- [ ] New or updated tests added
- [ ] Existing tests still pass (`uv run pytest`)
- [ ] Manually verified in a Jupyter notebook

## Checklist

- [ ] `trace_step` still returns the original output unmodified
- [ ] No deep copies of DataFrames introduced
- [ ] All new `datetime` fields use `datetime.now(timezone.utc)`
- [ ] No object references between Pydantic models (FK-only)
- [ ] PR is scoped to a single logical change
