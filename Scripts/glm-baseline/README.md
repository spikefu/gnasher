# GLM-5.3-Flash expert-streaming baseline

`bench.sh <label> <cache> [dyn-slots] [extra llama.cpp args]` runs
`llama-completion` from `~/Developer/llama.cpp-moe-stream/build/bin` against
the Unsloth UD-IQ4_XS GGUF in `~/models/GLM-5.3-Flash-GGUF/UD-IQ4_XS/` with
MoE expert streaming, and writes `results/<label>.log`. `run-matrix.sh`
reproduces the six-configuration matrix recorded in
[docs/GLM53_FLASH_FEASIBILITY.md](../../docs/GLM53_FLASH_FEASIBILITY.md).

`<cache>` is `Ns` for N slots per layer or a GiB budget. `dyn-slots` sets
`LLAMA_MOE_STREAM_DYN`, the size of the hotness-ranked pool; slots beyond it
are pinned by expert ID (the branch default is 3 x n_expert_used = 24).
Add `--moe-stream-direct` to bypass the macOS page cache.
