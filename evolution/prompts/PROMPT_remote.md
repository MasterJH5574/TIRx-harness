### Benchmark server (kcoral)

Score with the prescribed bench command and `--remote {{bench_server_url}}`.
Server startup belongs to another process. Before use or after disconnection:
```bash
until curl -fsS --max-time 5 {{bench_server_url}}/health; do sleep 5; done
```
Require `status: ok`; otherwise keep polling. Health lists `gpus[].gpu_id`,
`workers[].status` (`idle`/`busy`), and `queue_length`. Never start, stop, kill,
restart, or reconfigure the server. Use its provisioned environment in all
CLIs and uploaded scripts: no package installs/upgrades/removals, environment
setup scripts, server configuration edits, or shared-file changes. Report
missing dependencies; do not repair them. Continue with available tools.
Runners use the server's user, tools, and assigned GPU, without a filesystem
sandbox. This access does not authorize environment changes.

### Every GPU run goes to the server

Scoring and {{remote_gpu_work}} must use the same server. Never run candidates,
baselines, profilers, or CUDA work on local GPUs; local evidence is invalid.
Dropping `--remote` to score in-process invalidates the run.
{{remote_local_checks}}

Run examples from `{{worktree}}`. Supply candidates, scripts, harness files,
and inputs with every request; never rely on previous requests. Uploads go to
fresh temporary working directories (`$KCORAL_DIR`), deleted after execution.
Keep explicit scratch/output writes there. Save results locally with your
artifacts. Stdout/stderr print locally on completion. Keep CPU-only inspection,
report/trace postprocessing, and output filtering local.

Dedicated CLIs allow repeatable `--send` and `-e NAME=VALUE`; `-e NAME` forwards
its local value. Overrides apply to this request's subprocess only; do not
change GPU assignment or persistent configuration. Keep their 300-second
timeout unless the run is known to need longer.
Tool/application arguments are forwarded to remote execution; wrapper options
control uploads, environment, timeouts, and local artifact retrieval.

Use dedicated CLIs for the tools below. For other GPU or kernel diagnostics,
use `--cmd`; it returns only stdout/stderr (up to 16 MiB), so print results:
```bash
python evolution/remote/kcoral_remote.py --remote {{bench_server_url}} \
  --send <input-dir> --cmd '<bash script>'
```

### Dedicated tools

```bash
# NCU: mandatory -o is local; wrapper/NCU options precede --, target follows.
python evolution/remote/kcoral_ncu.py --remote {{bench_server_url}} \
  --send <input-dir> -o <local-report.ncu-rep> \
  --set full --launch-count 1 -- python capture.py

# IKET: mandatory --output-dir is local; output files return automatically.
python evolution/remote/kcoral_iket.py --remote {{bench_server_url}} \
  --send <input-dir> --output-dir <local-output-dir> \
  -- profile --postprocess json -- python capture.py

# Compute Sanitizer: --error-exitcode 1 makes findings fail the command.
python evolution/remote/kcoral_compute_sanitizer.py \
  --remote {{bench_server_url}} \
  --send <input-dir> -- --tool memcheck --error-exitcode 1 python check.py

# Python: only stdout/stderr and exit status return; print results, not files.
python evolution/remote/kcoral_python.py --remote {{bench_server_url}} \
  --send <input-dir> -e MODE=test -- check.py
```
Use Python NCU targets without shell commands or nested GPU profilers;
inspect locally with `ncu --import`. For IKET, use established annotations.
To retrieve a sanitizer `--log-file check.log`, add wrapper options
`--fetch check.log --output-dir <local-output-dir>`.
Python accepts a script, `-c 'code'`, or `-m module` after `--`, without a
`python` prefix. Read/import uploaded inputs; avoid output writes.
