# Maintaining the documentation site

The site uses Sphinx, MyST Markdown, and the Furo theme. See
[Build the docs](development/build-the-docs.md) for the complete build,
local preview, dependency update, and automated-check workflow.

From the repository root, the minimal build and preview commands are:

```bash
uv venv --python 3.12 .local/docs-venv
uv pip install --python .local/docs-venv/bin/python -r docs/requirements.txt
.local/docs-venv/bin/python -m sphinx -E -a -b html -n -W --keep-going docs docs/_build/html
.local/docs-venv/bin/python -m http.server 8018 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://127.0.0.1:8018>. The build needs no kernel runtime or initialized
submodules. Stop the server with Ctrl+C and rebuild after editing pages.
