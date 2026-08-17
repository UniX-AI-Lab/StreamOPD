# verl (vendored)

This is a copy of [verl](https://github.com/volcengine/verl) 0.8.0.dev, Apache-2.0, carrying
the on-policy distillation work that StreamOPD builds on. Upstream documentation lives at
<https://verl.readthedocs.io>.

Install it in editable mode from the repository root:

```bash
pip install -e ./verl --no-build-isolation
```

The changes relative to upstream are listed in [PATCHES.md](PATCHES.md). They are all gated
behind config flags that default to off, so with the flags unset this behaves as upstream
verl does.

Upstream docs, examples, tests and CI configuration were removed to keep this repository
focused; fetch them from the upstream project if you need them.
