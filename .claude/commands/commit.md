---
description: "Commit staged and unstaged changes with automatic git add && git commit"
---

Create git commits by automatically staging files and committing them. If changes span multiple logical areas, split into separate commits. Otherwise, create a single commit. Use only "Generated with Claude" as attribution.

Please run git status and git diff to understand the current changes, then:

1. Analyze the changes to determine if they should be split into multiple logical commits based on different functional areas
2. Stage and commit the changes using `git add <files> && git commit -m "commit message"`
3. Use only "Generated with Claude" as the attribution text - no other attribution text or links
4. Create descriptive commit messages that explain what was changed and why

If provided arguments, use them as the commit message: $ARGUMENTS