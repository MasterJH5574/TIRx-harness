# TIRx Harness documentation website

Generated documentation for [TIRx Harness](https://github.com/mlc-ai/TIRx-harness),
served at <https://tirxharness.mlc.ai/docs/>.

This public repository contains the generated website. The source repository is
private. The source repository's Documentation workflow builds and synchronizes
this repository after updates to `main`. Edit documentation in TIRx-harness;
the next successful synchronization replaces generated files here.
`source.json` identifies the source commit used for this build.

## Hosting

GitHub Pages serves branch `main` at `/` with the custom domain
`tirxharness.mlc.ai`, also recorded in `CNAME`. The root `index.html` redirects
to `docs/`; `.nojekyll` preserves Sphinx's underscored asset directories.

In the `mlc.ai` DNS settings, the `CNAME` alias named
`tirxharness` points to `mlc-ai.github.io`. Enable **Enforce HTTPS** in Pages
settings after GitHub issues the domain's certificate.

Only generated documentation, its page sources, and static assets are
synchronized here. Kernel implementations and the source repository's Git
history are not copied. The publication key is scoped to this repository.

## Local preview

Serve the repository root:

```bash
python -m http.server 8018 --bind 127.0.0.1 --directory .
```

Open <http://127.0.0.1:8018/docs/>.
