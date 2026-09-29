# External implementation repositories

The repository names and HTTPS URLs are defined once in
[sources.json](../sources.json). The
[skill fetcher](../../scripts/fetch_references.py) shallow-clones each
repository's current default branch into this directory. Eval setup invokes
the same fetcher and excludes repositories wholly covered by the task's banned
paths.

These are deliberately live references, not reproducible dependencies. Before
using a checkout to explain version-sensitive source behavior, inspect the
source itself and state that the finding applies to the revision present in the
current eval worktree.

`tirx-kernels/` provides the complete kernel source and root README for wiki
lookup. Pip separately installs the latest default-branch `tirx_kernels` into
each fresh venv for execution.
