#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// The cache is partitioned: its low slots permanently hold the experts of the same ids and are
// never evicted, the remaining slots form the dynamic pool the wave planner and the remap draw
// from. A host mirror holds the experts the partition does not cover, so their misses cross PCIe
// instead of reading the model file, and no expert is stored twice.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab

    const uint8_t * host       = nullptr; // pinned host mirror of the unpinned experts, null when unmirrored
    uint32_t        host_first = 0;       // first expert the mirror holds

    std::string src_name;      // GGUF tensor name, matched against the expert pack
    size_t      pack_off = 0;  // byte offset of this weight's slab inside a packed expert blob
    bool        direct_write = false; // the cache buffer is host-writable: pread straight into the slot
};

struct llama_moe_stream_layer;

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;
    uint32_t n_pinned = 0; // slots 0..n_pinned-1 permanently hold experts 0..n_pinned-1

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // expert pack: all weights of one expert contiguous at pack_base + expert*pack_stride
    bool     packed      = false;
    uint64_t pack_base   = 0;
    uint64_t pack_stride = 0;

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_claimed;  // [n_slots] a worker owns the load
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // lookahead prefetch: this layer's router (set after load), and the experts an earlier layer's
    // remap predicted for it, one record per lookahead distance (guarded by mgr->mtx)
    ggml_tensor * gate_inp    = nullptr;
    ggml_tensor * exp_probs_b = nullptr;
    struct pred_record {
        std::vector<int32_t> ids;     // predicted ids, rank-major per token
        std::vector<int32_t> fetched; // experts a prefetch was issued for
        int32_t n_per_token = 0;
        bool    valid       = false;
    };
    std::vector<pred_record> pred; // [lookahead] index d-1

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0; // stale unless it matches slot_gen[slot]
    int64_t  t_push_us = 0; // when the remap queued it
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    uint32_t n_pinned     = 0; // slots holding the resident expert partition, never evicted
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct, uint64_t ram_budget);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    // partition the experts: the low ids stay resident in the cache slots, the rest are mirrored
    // in pinned host memory and cross PCIe on demand, so no expert is stored twice
    void pin_partition(uint32_t n_expert_used);
    void fill_host_mirror();

    size_t size_bufs() const;

    void print_stats() const;

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    // optional expert pack sidecar (<first gguf>.epack or LLAMA_MOE_STREAM_PACK): one blob per
    // (layer, expert) holding all of its slabs, so a miss is one contiguous read
    int         pack_fd = -1;
    std::string pack_path;
    void open_pack(const std::string & path);
    bool pack_enabled() const { return pack_fd >= 0; }

    // whether a cache buffer can be written through its host pointer (CPU buffers and Metal shared
    // buffers); verified once per buffer at open time
    void detect_direct_write();

    size_t ram_budget = 0; // pinned host mirror budget in bytes
    std::vector<ggml_backend_buffer_ptr> host_bufs;

    size_t  max_nb_expert      = 0;
    size_t  max_nb_blob        = 0; // largest packed expert blob (sum of its slabs)
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;
    std::deque<llama_moe_stream_work> q_prefetch; // lookahead loads, served when q_demand is empty

    // lookahead prefetch (LLAMA_MOE_STREAM_LOOKAHEAD=D, default 1, 0 disables): layer L's remap also
    // receives the routing that layers L+1..L+D's routers produce on L's input (a prediction of their
    // real routing); the predicted experts that are not resident are prefetched while L's GEMMs and
    // L+1's attention run. measured on GLM-5.3-Flash: 68% of the distance-1 predictions are confirmed
    // (98% at rank 1, 35% at rank 8), decode +6%; distance 2 and rank caps measured no better
    int32_t lookahead      = 1;
    int32_t lookahead_k[4] = {};   // predicted ranks to prefetch per token, per distance (0 = all n_expert_used)
    bool    lookahead_stat = false; // LLAMA_MOE_STREAM_LOOKAHEAD_STAT=1: measure accuracy, no prefetch

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;

    struct {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t t_stall_us  = 0; // wait time in miss handling
        int64_t t_read_us   = 0; // worker time spent inside the file reads (pure I/O)
        int64_t n_read      = 0; // expert loads performed by workers
        int64_t t_handoff_us = 0; // remap push -> worker starts the read

        int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload
        int64_t t_stall_wave_us  = 0; // wait time in wave miss handling

        // lookahead prefetch, per distance d (index d-1, up to 4)
        int64_t pred_n[4]           = {}; // predictions scored
        int64_t pred_hit[4]         = {}; // predictions the real routing confirmed
        int64_t pred_rank_n[4][16]  = {}; // per predicted rank
        int64_t pred_rank_hit[4][16]= {};
        int64_t pred_miss[4]        = {}; // real demand misses in layers that had a prediction
        int64_t pred_miss_covered[4]= {}; // ... that the prediction contained
        int64_t pred_fetch[4]       = {}; // prefetch loads issued
        int64_t pred_fetch_used[4]  = {}; // ... whose expert the real routing then needed
    } stats;

    // internals
    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
// same op under a distinct function pointer, never registered as a GPU host op: measured faster for
// single-token decode, where the scheduler split beats a GPU-side event wait on Apple Silicon
void llama_moe_stream_remap_decode(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// the remap with lookahead: b holds the predicted ids [n_expert_used, n_tokens, D] of layers il+1..il+D;
// predicted misses are prefetched (or just scored under _STAT)
void llama_moe_stream_remap2(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
void llama_moe_stream_remap2_decode(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:  makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask: emits 1.0 for pairs belonging to wave w, 0.0 otherwise
void llama_moe_stream_wave_ids (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata);
