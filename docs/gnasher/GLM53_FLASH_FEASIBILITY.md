# GLM-5.3-Flash on TurboFieldfare principles: RAM sizing

Date: 2026-10-02. Sources: zai-org/GLM-5.3-Flash config.json, Unsloth GGUF
notes, llama.cpp PR #25294 (MoE expert streaming), oMLX PR #3865
(expert offload on a 48 GB M4 Pro), PipeNetwork/glm53-flash-mlx.

## Model shape (from config.json)

| Property | GLM-5.3-Flash | Gemma 4 26B-A4B (current target) |
| --- | ---: | ---: |
| Total / active params | 320.6B / 18B | 26B / 3.9B |
| Layers | 45 (3 dense + 42 MoE) | 30 (all MoE) |
| Routed experts per layer, active | 288, 8 (+1 shared) | 128, 8 (+1 shared) |
| Params per routed expert | 25.2M (3 x 4096 x 2048) | ~6M |
| Routed-expert params total | 304.4B | ~24B |
| Non-routed ("core") params | 16.2B | ~2.4B (1.35 GB at 4-bit) |
| Attention | 34 KDA linear-attention + 11 sparse MLA (kv_lora_rank 512, no RoPE) | 25 SWA + 5 global |
| KV per token | ~11 KiB (MLA only; KDA state is constant-size) | ~75 KiB |

The core is 13.4B of KDA/MLA/router weights, 1.06B shared experts, 0.45B
dense MLP, and 1.27B embedding + head (not tied).

## What must stay resident

| Component | 4-bit g64 | 6-bit | 8-bit / FP8 |
| --- | ---: | ---: | ---: |
| Core weights | 9.1 GB | 13.2 GB | 16.2-17.2 GB |
| KV cache (32K / 128K) | 0.37 / 1.48 GB | same | same |
| One routed-expert blob | 14.2 MB | 20.4 MB | 25-27 MB |
| Routed experts on disk | 171 GB | 247 GB | 304-323 GB |

The core alone rules out 8 GB Macs. 16 GB is possible only with a 4-bit core
and almost no expert cache.

## Per-token expert I/O (the speed limiter)

Each token touches 8 x 42 = 336 routed experts.

| Expert precision | Bytes/token at 0% cache hits | at 50% | at 70% | at 80% |
| --- | ---: | ---: | ---: | ---: |
| 4-bit | 4.76 GB (~1.1 tok/s at 5 GB/s) | 2.4 GB (~2.1) | 1.4 GB (~3.5) | 0.95 GB (~5.3) |
| FP8 | 8.46 GB (~0.6 tok/s) | 4.2 GB (~1.2) | 2.5 GB (~2.0) | 1.7 GB (~3.0) |

Internal Apple SSDs deliver roughly 4-7 GB/s on 14-27 MB reads, so the
ceiling is a few tokens per second even with a large cache. External
evidence: llama.cpp streaming GLM-5.2 (754B) got 73-79% hit rate with
55-79 GB of cache and ~2 tok/s decode; oMLX on a 48 GB M4 Pro ran a 153 GiB
MoE at ~1.6 tok/s with a 21.7 GiB footprint and ~50% hit rate. Routing is
skewed (top-32 of 256 experts cover ~78% of decode routes on MiMo), so a
cache well below full size still earns most of the hits.

## Expert-cache coverage per GB

| Cache RAM | Share of 4-bit experts | Share of FP8 experts |
| ---: | ---: | ---: |
| 8 GB | 5% | 3% |
| 16 GB | 9% | 5% |
| 32 GB | 19% | 11% |
| 64 GB | 37% | 21% |
| 90 GB | 53% | 30% |

## Recommended configurations by Mac RAM

Budget = core + KV + expert cache + ~4 GB for macOS/app/file cache.

| Mac RAM | Core | Experts on disk | Expert cache | Context | Expected decode | Verdict |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 16 GB | 4-bit (9 GB) | 4-bit (171 GB) | ~1-2 GB | 8K | <1 tok/s | Runs, barely; not useful |
| 24 GB | 4-bit | 4-bit | ~8 GB | 16K | ~1 tok/s | Marginal |
| **32 GB** | 4-bit core, attention 8-bit optional (9-13 GB) | 4-bit | ~12-16 GB | 32K | 1-2 tok/s | **Lowest useful spec** |
| 48 GB | 6-bit or 8-bit (13-17 GB) | 4-bit or 6-bit | ~25 GB | 64K | 2-3 tok/s | Comfortable; quality floor rises |
| 64 GB | 8-bit (17 GB) | 6-bit or FP8 | ~40 GB | 128K | 2-3 tok/s | Good quality/speed trade |
| 128 GB | 8-bit | FP8 native (304 GB, no requant) | ~95 GB (30% of FP8 experts) | 128K+ | 2-4 tok/s | Better than any fully-resident quant that fits |

For comparison, the fully-resident route on 128 GB is Unsloth UD-IQ3_XXS
(120 GB, 81.6% top-1 retention) or UD-Q2_K_XL (109 GB, 78.3%), and both
fight the Metal wired-memory limit. Streaming lets the 128 GB machine keep
experts at FP8 (lossless vs. the release) and pay only in tokens/sec.

## Disk

4-bit experts need ~180 GB free; FP8 experts need ~320 GB. A 512 GB SSD
works for the 4-bit pack only. External USB4 NVMe was fast enough in the
oMLX tests.

## Engineering gap vs. the Gemma 4 runtime

New Metal kernels: KDA (Kimi Delta Attention) recurrence, MLA with
compressed KV plus the k-pool sparse indexer (index_topk 2048), dense MLP
layers, mHC hyper-connections, and an untied head. The MoE streaming,
`.gturbo` repack, LFU slot cache, and chunked prefill carry over with
larger blobs (14-27 MB vs 3.4 MB) and 288 slots per layer file. Only the
mixed precision changes: keep experts at source precision where disk allows.

## Measured baseline (2026-10-02, this M5 Max 128 GB, internal SSD)

Setup: llama.cpp master (bed0a85, with GLM-5.3-Flash support merged
2026-09-30) plus the MoE expert-streaming branch from PR #25294 as rebased by
ServeurpersoCom, rebased again onto master, with three local commits
(`glm5next` architecture alias for the Unsloth GGUFs, F_NOCACHE so
`--moe-stream-direct` really bypasses the macOS page cache, and a
`LLAMA_MOE_STREAM_DYN` knob for the dynamic pool size). Checkout:
`~/Developer/llama.cpp-moe-stream`, branch `moe-stream-master`.
Model: unsloth/GLM-5.3-Flash-GGUF UD-IQ4_XS (146 GiB). Prompt 480 tokens,
256 generated, 8K context, temperature 0, Metal, 9 I/O threads.
Runner and logs: `scripts/gnasher-bench/`.

Fixed costs measured at load: 8.4 GB resident core on Metal, 0.64 GB
embeddings on CPU, 154 MB KV + indexer cache at 8K, 372 MB compute buffer.
Each expert slot is 11.3 MiB at IQ4_XS; 42 streamed layers.

| Config | Expert cache | Peak footprint | Hit rate | Prefill | Decode | Emulates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 24 slots/layer dynamic, buffered | 11.4 GB | 22.8 GB | 49.6% | 24.5 tok/s | 3.09 tok/s | 32 GB Mac, optimistic |
| 24 slots/layer dynamic, direct | 11.4 GB | 22.7 GB | 49.6% | 26.2 tok/s | 2.69 tok/s | 32 GB Mac, honest |
| 48 slots dynamic, direct | 22.8 GB | 34.8 GB | 61.9% | 35.2 tok/s | 3.69 tok/s | 48 GB Mac |
| 48 slots: 24 pinned by ID + 24 dynamic, direct | 22.8 GB | 35.3 GB | 54.8% | 26.3 tok/s | 3.21 tok/s | policy comparison |
| 96 slots dynamic, direct | 45.7 GB | 58.8 GB | 75.5% | 44.4 tok/s | 4.94 tok/s | 64 GB Mac |
| 160 slots dynamic, direct | 76.1 GB | 90.1 GB | 85.3% | 45.2 tok/s | 7.46 tok/s | 128 GB Mac |

All six runs produced byte-identical text, so streaming and cache size affect
latency only. The answer itself was coherent, did the arithmetic correctly
and followed the five-part structure.

Findings:

- The 32 GB-class configuration is genuinely usable: 2.7 tok/s decode with a
  22.7 GB footprint, leaving ~9 GB for macOS and apps on a 32 GB machine.
- Effective expert read rate was 2.3 GB per token at 24 slots in 372 ms,
  about 6 GB/s from the internal SSD. The SSD is slightly faster than the
  5 GB/s assumed above.
- Hotness-ranked eviction beats pinning by expert ID at equal RAM: 62% vs 55%
  hit rate and 3.7 vs 3.2 tok/s. Routing is skewed, but not toward low IDs.
- Hit rate grows roughly linearly with log(cache): 50% at 8% of experts,
  62% at 17%, 76% at 33%, 85% at 56%. A 128 GB Mac at 7.5 tok/s is already
  pleasant; experts at FP8 would halve that to a still-usable ~3.5 tok/s.
- Buffered reads on a machine with spare RAM inflate decode by ~15%; use
  `--moe-stream-direct` when emulating smaller Macs.

Repro:

```bash
cd scripts/gnasher-bench && ./bench.sh 24s-direct 24s "" --moe-stream-direct
```

## Speculative decoding under streaming (measured 2026-10-02)

The MTP draft head for GLM-5.3-Flash (llama.cpp PR #27917) was ported onto the
streaming branch (`glm5-mtp-stream` in `~/Developer/llama.cpp-moe-stream`).
Server runs, raw completion endpoint, same 480-token prompt, 256 generated,
160 slots per layer, buffered reads, page cache warm:

| Draft length | Acceptance | Mean accepted run | Decode |
| ---: | ---: | ---: | ---: |
| none | | | 7.33 tok/s |
| 1 | 77.1% | 1.77 | 7.28 tok/s |
| 2 | 66.5% (81.7%, 51.4% by position) | 2.33 | 7.11 tok/s |
| 4 | 38.3% (75%, 44%, 23%, 11%) | 2.52 | 5.65 tok/s |
| 6 | 26.9% | 2.60 | 4.53 tok/s |
| n-gram (`ngram-mod`, 4) on the same task, user-reported | | | 9.19 vs 8.6 tok/s |

All runs produced byte-identical text, so the trunk, its KDA rollback and the
draft head are consistent. First-position acceptance of 77-82% on dense
prose with arithmetic says the head is working; the PR author's 92% was
measured on other content.

Why it does not help here: a verification step over 1+k tokens touches the
union of their experts, and distinct tokens share few of the 8-of-288 routes,
so streamed expert I/O grows almost linearly with k. Measured: one token costs
141 ms, a 5-token verify step 447 ms, for 2.5 tokens accepted on average.
Speculative decoding pays off when weights are read once per step from
resident memory; prefill's 6x gain comes from hundreds of tokens sharing
experts, which a 4-token draft cannot replicate. On a machine where the
model is fully resident the same head should give the PR's 15-30%.

Consequence for the TurboFieldfare port: do not budget for speculative
decoding as a decode accelerator under expert streaming. The I/O-side levers
(one read per expert, higher queue depth, decode-reserved cache) remain.

## Cheap performance levers (measured 2026-10-02, IQ4_XS, buffered reads)

| Experiment | Setting | Hit rate | Decode | vs 160-slot baseline |
| --- | --- | ---: | ---: | ---: |
| I/O threads | 9 (auto) / 14 / 18 at 160 slots | 85.3% | 7.01 / 6.91 / 7.02 tok/s | no effect |
| Cache size | 176 slots, 86 GB | 87.2% | 7.91 tok/s | +13% |
| Cache size | 200 slots, 97 GB | 89.5% | 9.65 tok/s | +38% |
| Cache size | 216 slots, 105 GB | | fails: Metal command buffer error (status 5) during decode | GPU memory ceiling |
| Parallel streams | 4 requests, 160 slots | | 3.09 tok/s each, 12.4 tok/s aggregate | +77% throughput, 2.3x latency |
| Parallel streams, prefill | 4 x 486 tokens | | 81 tok/s aggregate | 2x |

Notes:

- Reader threads do not matter at this hit rate: the SSD is not queue-depth
  limited on 11 MiB reads, so the ~95 ms of I/O per token is bandwidth.
- Cache size is the strongest lever. 200 slots (97 GB cache + 9 GB core)
  loaded and ran on the 128 GB machine without raising the Metal wired limit;
  216 slots loaded but failed during decode with a Metal command-buffer error,
  so ~200 slots is the practical ceiling at IQ4_XS on 128 GB. Each
  4 points of hit rate removed ~20% of the I/O time.
- Parallel streams share expert reads the way prefill does. Aggregate
  throughput rises 1.8x with four streams, but each stream slows to 3.1
  tok/s, so this suits agent fan-out rather than interactive chat.
- Two-drive striping was not testable: the only external drive here is a
  USB stick reading at 183 MB/s.

## Direct slot writes and the expert pack (measured 2026-10-02)

Two runtime changes on the `glm5-mtp-stream` branch:

- **Direct-to-slot reads.** Metal shared buffers are host memory, so a miss is
  now `pread` straight into the cache slot instead of into a staging buffer
  followed by a copy. Verified per buffer at open by a write/read round trip.
  Kept off for unpacked reads under F_NOCACHE, where the 4 KiB alignment split
  cost more than the copy it saved.
- **Expert pack.** `scripts/moe_expert_pack.py` writes `<gguf>.epack`: one
  page-aligned blob per (layer, expert) holding gate, up and down together.
  A miss becomes one contiguous `preadv` instead of three reads at three file
  offsets. GLM-5.3-Flash IQ4_XS: 147 GB, built in about 3 minutes.

Same prompt, 200 slots, byte-identical output in every run:

| Model / mode | Unpacked, staging | Packed, direct write | Gain |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B Q8, 24 slots, uncached (SSD-bound) | 6.5-6.7 tok/s | 10.4 tok/s | +57% |
| Qwen3.5-35B-A3B Q8, 24 slots, buffered (file-cache-bound) | 18-20 tok/s | 19.3 tok/s | none |
| GLM-5.3-Flash IQ4_XS, 200 slots, buffered | 9.8-9.9 tok/s | 11.1-11.3 tok/s | +14% |
| GLM-5.3-Flash IQ4_XS, 200 slots, uncached | 10.1 tok/s | **12.5 tok/s** | +24% |

Notes:

- Whether a run is SSD-bound depends on what the file cache already holds.
  F_NOCACHE still serves cached pages, so a GGUF just read by the pack builder
  benchmarks far faster than a cold one. The Qwen "uncached" unpacked figure
  swung between 6.5 and 16 tok/s across sessions for this reason. GLM at 147 GB
  of experts plus a 107 GB wired cache never fits, so its numbers are stable.
- With the pack, uncached beats buffered on GLM: aligned whole-blob reads gain
  nothing from the page cache and skip its copy.
- Direct writes alone are worth 1-3%. The pack is the lever.

Recommended launch on this machine (12.5 tok/s decode, 46 tok/s prefill):

```sh
LLAMA_MOE_STREAM_DYN=200 llama-server -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
    --moe-stream-cache 200s --moe-stream-ram 0 --moe-stream-direct -c 65536 \
    --chat-template-kwargs '{"reasoning_effort":"high"}' --port 8080
```

The pack beside the GGUF is picked up automatically; `LLAMA_MOE_STREAM_PACK`
overrides the path. Progress today: 7.5 -> 9.65 (cache 200 slots) -> 12.5
tok/s (pack + uncached), with the trunk output unchanged.


## Metal host-callback op for the remap (measured 2026-10-02, negative result)

Hypothesis: the 42 per-token graph splits at the expert-id remap (a CPU custom
op) cost ~0.5 ms each in Metal command-buffer drain and re-encode, ~20 ms per
token. Implementation: the Metal backend claims registered custom ops and
runs them mid-command-buffer via a pair of shared events (GPU signals when the
router output is ready, a host thread runs the remap, GPU waits on the second
event). Two host-side wake-up variants: MTLSharedEventListener, and a thread
that spin-polls the event value. Code: `ggml_metal_device_host_op_*` in the
Metal backend, `ggml_backend_metal_register_host_op`, opt-in from llama with
`LLAMA_MOE_STREAM_METAL_HOST_OP=1`.

Result, same prompt, 256 tokens, alternating runs in one session:

| Model, cache | CPU split (default) | Host op, listener | Host op, spin-poll |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B Q8, 255 slots (all hits) | 20.4-29.0 ms/token | 25.0-27.8 | 24.6 |
| Qwen3.5-35B-A3B Q8, 24 slots packed | 49-54 ms/token, prefill 131-193 tok/s | 68 ms, prefill 199 | 50-56 ms, prefill 137-208 |
| GLM-5.3-Flash IQ4_XS, 200 slots packed | 79.1 ms/token, prefill 44-46 | 79.2, prefill 46.5 | 78.3-80.1, prefill 46.7 |

The GPU idling at an encoded event wait and resuming costs about what the
scheduler's split-and-resubmit costs, and the host-side wake-up mechanism
makes no difference (listener and spin-poll tie). On GLM the change is
invisible either way, which also shows the earlier "20 ms of split overhead"
estimate was wrong: GLM's remaining ~65 ms per token is compute. Each mode is
deterministic with itself but the three graph shapes (resident, split,
unsplit) produce slightly different rounding and therefore different text at
temperature 0 after a couple of hundred tokens; none is more correct.

Kept as opt-in for experimentation. The default remains the scheduler split.
Lesson for the Swift port: owning the command stream does not remove this
cost; the CPU has to learn the routing before the expert matmul, and any
GPU<->CPU handoff on Apple Silicon costs a few hundred microseconds per layer.
The only way around it is to not need the CPU per layer, which means
GPU-resident routing tables with misses handled some other way.

## Where GLM-5.3-Flash decode time goes (measured 2026-10-02)

Tools added: `GNASHER_PROFILE_OPS=1` in `llama-completion` (per-op time via the
scheduler eval callback; every node is synchronized, so a ~155 us floor is
subtracted) and `GNASHER_PERF_SHAPES=1` for `test-backend-ops perf`
(expert matmul at GLM shapes, 288 experts, 8 used).

Expert matmul kernel speed by quant type, Metal, n=1 (cache-resident microbenchmark,
so absolute numbers are optimistic; relative order is what matters):

| type | gate/up (k=4096, m=2048) | down (k=2048, m=4096) | bits/weight |
| --- | ---: | ---: | ---: |
| q4_0 | 41.7 us | 40.5 us | 4.5 |
| iq4_xs (current) | 57.8 us | 51.0 us | 4.25 |
| q4_K | 43.9 us | 75.0 us | 4.5 |
| q3_K | 56.4 us | 75.8 us | 3.4 |
| iq3_xxs | 71.6 us | 73.7 us | 3.06 |
| q6_K | 81.1 us | 104.3 us | 6.6 |
| q5_K | 98.4 us | 72.5 us | 5.5 |
| q8_0 | 120.7 us | 119.1 us | 8.5 |

The expert matmuls are bandwidth-bound: q8_0 at twice the bytes takes about
twice as long, and q4_0's 25% kernel advantage over iq4_xs comes with 6% more
bytes, so requantizing the experts is not a lever. In the real run the three
expert matmuls cost ~21 ms per token for 4.7 GB of weights, ~224 GB/s.

Decode sensitivity to Metal graph features (200 slots, packed, uncached, short
prompt so the cache starts cold): default 110-113 ms/token, fusion disabled
113, concurrency disabled 112. Neither matters, so the ~7,700 graph nodes per
token are not a measurable encode cost either.

Budget per token at 200 slots on the 480-token prompt (79 ms): ~26 ms I/O
stall inside the remap, ~21 ms expert matmuls, ~32 ms everything else
(KDA/MLA attention, hyper-connections, router chain, shared experts, head).
The last bucket is the one still unprofiled at kernel level.

Per-op decode costs above the sync floor (480-token prompt, 128 generated,
200 slots packed uncached; floor 157 us from the no-op rows):

| op | calls/token | us above floor | ms/token |
| --- | ---: | ---: | ---: |
| MAP_CUSTOM1 remap (includes expert I/O wait) | 42 | 961 | 40.1 |
| MUL_MAT_ID ffn_moe_gate | 42 | 268 | 11.4 |
| MUL_MAT_ID ffn_moe_up | 42 | 198 | 8.4 |
| MUL_MAT_ID ffn_moe_down | 42 | 167 | 7.1 |
| MUL_MAT kda_out | 34 | 96 | 3.3 |
| MUL_MAT ffn_gate / ffn_up (per-layer dense) | 45 each | 40-45 | 3.8 |
| MUL_MAT attn_out (MLA) | 11 | 166 | 1.8 |
| GLU swiglu (moe + dense) | 88 | 23-41 | 2.8 |
| MUL_MAT ffn_shexp | 42 | 27 | 1.1 |
| everything else | ~7,000 | < 3 each | below resolution |

The three expert matmuls read 4.7 GB per token in ~27 ms, about 175 GB/s,
while the cache-resident microbenchmark of the same kernel implies several
times that. They are the single largest compute item and the only one worth
kernel work. `xctrace` is not available here (Command Line Tools only), so
there is no GPU timeline; the next check is a DRAM-bound microbenchmark of
MUL_MAT_ID against a plain MUL_MAT of equal bytes.

## Why the expert matmuls run below bandwidth in situ (measured 2026-10-02)

Microbenchmarks cleared the kernel: at DRAM-bound sizes MUL_MAT_ID equals a
plain MUL_MAT of the same bytes and every quant type lands at 540-595 GB/s,
including experts scattered over a 1.4 GB tensor. Residency sets on or off
make no difference. Yet in the streamed run the three expert matmuls reach
~175 GB/s, and the same slowdown appears on the streamed Qwen model whose
experts are tiny, with the same fingerprint in both: the first matmul after
the remap is the slowest, the second faster, the third fastest.

That is a GPU clock ramp. Every layer the GPU idles for a fraction of a
millisecond while the CPU remaps (and the host-op experiment showed an
in-command-buffer event wait idles it just the same), it drops frequency, and
the next kernels run slow while it recovers.

Test: `GGML_METAL_KEEP_WARM=1` starts a thread that submits 64 KB blit fills
on its own queue so the GPU never idles (`GGML_METAL_KEEP_WARM_US=N` pauses
between fills). GLM-5.3-Flash, 200 slots, packed, uncached, alternating runs:

| mode | ms/token |
| --- | ---: |
| plain | 82.5, 81.0, 80.2 |
| keep-warm continuous | 74.6, 78.3, 71.6 |
| keep-warm, 50 us pause | 76.7 |
| keep-warm, 200 us pause | 77.6, 78.7 |

A 4-8% gain with the I/O stall unchanged, so it is all compute. Kept opt-in
because it holds the GPU busy continuously; a version that pulses only during
the remap gaps would keep most of the gain at a fraction of the power. A
heavier concurrent load (test-backend-ops SCALE loop) made things slower, so
the benefit is specifically from avoiding idle, not from contention.

## State of play

GLM-5.3-Flash IQ4_XS on this M5 Max: 7.5 tok/s this morning, 12.5-14 tok/s
now (200 slots, expert pack, uncached reads, optionally keep-warm), identical
output throughout. Remaining per-token budget at 200 slots: ~26 ms expert I/O
stall (hit rate 89.5%, RAM-bound), ~21-27 ms expert matmuls (bandwidth plus
clock ramp), ~25-30 ms of small kernels across ~7,000 nodes. What would move
each: more RAM or lower-precision experts for the first; keep-warm or pulsed
warm-up for the second; a kernel-level GPU timeline (needs Xcode's xctrace,
not installed) to find fusion targets for the third. Speculative decoding,
Metal host ops, I/O thread counts, quant-type kernels and residency were all
measured and found not to help here.

## Pulsed keep-warm (measured 2026-10-02)

`GGML_METAL_KEEP_WARM=pulse`: the keep-warm thread fills only while the MoE
streaming remap holds a pulse, plus a linger after release
(`GGML_METAL_KEEP_WARM_LINGER_US`, default 1000). The GPU is left completely
idle between requests, unlike continuous mode. GLM-5.3-Flash, 200 slots,
packed, uncached, alternating runs:

| mode | ms/token |
| --- | ---: |
| plain | 79.5, 80.0, 82.2 |
| pulse, linger 300 us | 80.3, 80.2 (no gain) |
| pulse, linger 600 us | 72.5 |
| pulse, linger 1000 us (default) | 72.2, 72.5, 72.6 |
| pulse, linger 2000 us | 75.0 |
| continuous | 71.8, 75.7 |

The remap alone is not the idle window: the scheduler drains the GPU before
the remap and resubmits after it, so the fills must span about a millisecond
past the remap's return. 600-1000 us is the knee; longer lingers start to
contend. Result: ~9% on decode, 12.5 -> 13.8 tok/s, output unchanged, and no
GPU activity when nothing is being generated.

Recommended launch on this machine now adds the pulse:

```sh
GGML_METAL_KEEP_WARM=pulse LLAMA_MOE_STREAM_DYN=200 llama-server \
    -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
    --moe-stream-cache 200s --moe-stream-ram 0 --moe-stream-direct -c 65536 \
    --chat-template-kwargs '{"reasoning_effort":"high"}' --port 8080
```

## GPU timeline (Metal System Trace via xctrace, 2026-10-02)

With Xcode installed, `scripts/gnasher-bench/gpu-trace.sh` records a Metal
System Trace of a short decode and `xctrace_parse.py` reads the exported
tables. GLM-5.3-Flash, 200 slots, packed, uncached, pulsed keep-warm:

- During decode the GPU is **Active 20-25% of the time** (metal-gpu-state-intervals:
  ~200-260 ms active per second). The llama process's "GPU Execution"
  intervals total ~17 ms per generated token.
- Between consecutive llama GPU intervals the median gap is 235 us, p90 317 us,
  and these 100-700 us gaps account for essentially all of the idle time:
  roughly 1.4 ms of CPU-side time per streamed layer against ~0.4 ms of GPU work.
- Of the per-token budget (~75 ms here) that puts GPU compute at ~17 ms, the
  measured in-remap I/O stall at ~26 ms, and ~30 ms of other CPU-side work
  per token: scheduler split synchronization, input copies, Metal command
  encoding (serialized with execution because every split waits for the
  previous one), and remap bookkeeping.

This supersedes the eval-callback profile, whose per-node synchronization
attributed CPU-side waiting to GPU ops. The expert matmuls are not slow; the
GPU is mostly waiting. The next measurement is a timer inside the scheduler's
split loop (`GGML_SCHED_PROFILE=1`) to apportion the ~30 ms.

## Scheduler split-loop profile (GGML_SCHED_PROFILE=1, 2026-10-02)

Decode token at 200 slots, packed, uncached, pulsed keep-warm: 73.8 ms inside
the scheduler's split loop, 108 splits per token (54 Metal, 54 CPU).

| bucket | ms/token | notes |
| --- | ---: | --- |
| CPU waiting for the GPU before each CPU split | 40.1 | GPU execution is ~17 ms (trace); the other ~23 ms is launch/complete/wake latency over 54 splits, ~0.43 ms each |
| CPU compute (the remap) | 29.2 | of which 28.5 ms is the in-remap I/O stall; the remap's own work is <1 ms |
| Metal graph_compute (encode + submit) | 4.4 | encoding is not the problem |
| input copies | 0.1 | |

The 54 Metal splits instead of 42: the lightning indexer op of the 11 sparse
attention layers has no Metal implementation and runs on the CPU, so each of
those layers costs an extra GPU round trip and its own CPU compute.

Levers this exposes, with the ms they address:
1. I/O stall (28.5): a miss is one 11.3 MB read on one thread; most layers have
   0-2 misses, so the other reader threads sit idle. Reading a single expert as
   several parallel chunks cuts the per-miss latency roughly by the chunk count.
2. Round-trip latency (~23): fewer splits (move the indexer to Metal: -11 per
   token) and a cheaper wait (spin on command-buffer status briefly before
   blocking, to avoid the thread wake-up on every split).
3. Encode (4.4): already small.

## Three per-layer levers, measured (2026-10-02)

GLM-5.3-Flash, 200 slots, packed, uncached, pulsed keep-warm, 128 tokens
after the 480-token prompt, alternating runs:

| change | splits/token | ms/token | verdict |
| --- | ---: | ---: | --- |
| none | 108 | 75.3, 75.0 | |
| lightning indexer on Metal (32 heads) | 86 | 72.2, 72.4 | **-4%, default on**; text identical to the CPU indexer |
| chunked reads, 4 per miss | 108 | 74.2 | within noise; kept (default 4) |
| spin 1.5 ms before the blocking GPU wait | 108 | 81.1 | worse; off |
| indexer + chunks + spin | 86 | 73.3 | |

The indexer change is one runtime argument: the Metal kernel had its head
count hard-coded to DeepSeek V4's 64; GLM's indexer has 32 heads and fell back
to the CPU, costing a GPU round trip per sparse-attention layer.

Why chunked reads do not help in situ: decode-only statistics (short prompt)
show the remap-to-worker handoff at 0.026 ms per load and the read itself at
1.71-1.79 ms per 11.3 MB expert for 1, 4 or 8 chunks alike, i.e. ~6.6 GB/s.
That is the SSD's sustained rate; an isolated microbenchmark looked 2-3x
faster only because recent runs had left blobs in the page cache (F_NOCACHE
still serves cached pages). The I/O term is bandwidth-bound: fewer misses
(RAM), fewer bytes (precision) or more drives are the only ways down.

Remaining per-token budget at 200 slots (~72 ms): ~28 ms SSD stall, ~17 ms GPU
execution, ~20 ms GPU launch/complete/wake latency across 43 Metal splits,
~4 ms encode. The latency bucket is now the largest soft target; the only
structural fix is fewer or cheaper CPU<->GPU round trips, which the host-op
experiment showed the Metal event mechanism cannot provide.
