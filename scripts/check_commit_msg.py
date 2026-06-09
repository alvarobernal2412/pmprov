"""
Commit-msg hook: enforces Conventional Commits format and a 45-character
subject-line limit.

Usage (called by pre-commit):
    python scripts/check_commit_msg.py <commit-msg-file>
"""
import re
import sys

CONVENTIONAL = re.compile(
    r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)"
    r"(\([^)]+\))?!?: .+"
)
MAX_LEN = 45

msg_file = sys.argv[1]
with open(msg_file, encoding="utf-8") as fh:
    lines = [l for l in fh.read().splitlines() if not l.startswith("#")]

subject = lines[0].strip() if lines else ""

errors = []

if not CONVENTIONAL.match(subject):
    errors.append(
        f"  Not a conventional commit.\n"
        f"  Expected: <type>(<scope>): <description>\n"
        f"  Types: feat fix docs style refactor test chore perf ci build revert\n"
        f"  Got:      {subject!r}"
    )

if len(subject) > MAX_LEN:
    errors.append(
        f"  Subject too long ({len(subject)} chars, max {MAX_LEN}).\n"
        f"  Got: {subject!r}"
    )

if errors:
    print("Commit message rejected:\n" + "\n".join(errors))
    sys.exit(1)
