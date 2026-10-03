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
Runner and logs: `Scripts/glm-baseline/`.

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
cd Scripts/glm-baseline && ./bench.sh 24s-direct 24s "" --moe-stream-direct
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
