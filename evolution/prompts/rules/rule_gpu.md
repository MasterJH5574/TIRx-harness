GPU selection: this machine is shared and NO default GPU is set. Your
   subprocess starts with all GPUs visible. You MUST pick an idle one for
   every bench / ncu / correctness-GPU-run yourself:

       nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

   Pick an index with low utilization AND high free memory, then invoke with
   explicit override:

       CUDA_VISIBLE_DEVICES=<picked> python evolution/benchmark/adapter.py <workload_dir> vN
       CUDA_VISIBLE_DEVICES=<picked> ncu -k ... python evolution/benchmark/adapter.py <workload_dir> vN

   Bench termination: explicit PID/PGID ownership or `timeout` only — never
   fuzzy process-name matching.

   Re-check before each bench — a GPU that was idle 5 minutes ago may be
   busy now. If you skip this and just run `python evolution/benchmark/adapter.py ...` with no CVD,
   torch defaults to GPU 0, which may be busy — bench numbers will be
   garbage AND you'll cause contention for other users.
