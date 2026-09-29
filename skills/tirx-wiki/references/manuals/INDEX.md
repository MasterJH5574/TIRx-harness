# NVIDIA manuals

The upstream URLs are defined once in [sources.json](../sources.json). The
[skill fetcher](../../scripts/fetch_references.py), also used by eval setup,
fetches those pages without using a committed snapshot and materializes the
following searchable files under this directory:

```text
cuda_programming_guide.rst
ptx_isa.rst
ncu_cli_reference.rst
ncu_profiling_guide.rst
```

Search for the exact instruction, metric, option, or memory-ordering term
instead of reading a manual linearly. The fetcher extracts the upstream article
HTML and converts it with Pandoc, retaining document structure and remote image
references while dropping web chrome. Follow the `Source:` URL recorded before
each page when the original browser rendering matters.

NVIDIA renumbers sections between editions. Cite an anchor or instruction name,
never a section number or local line offset. Because each fetch uses the
currently served pages, two materializations created at different times may
contain different manual revisions.
