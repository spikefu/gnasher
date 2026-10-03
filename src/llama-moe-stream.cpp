#include "llama-moe-stream.h"

#include <atomic>
#include <chrono>
#include <functional>

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// demand loads a worker drains per pass; one stream sync amortizes over the whole batch
static const size_t MOE_STREAM_UPLOAD_BATCH = 16;
static const size_t MOE_STREAM_FILL_CHUNK   = 64ull*1024*1024; // bulk read size of the load-time fills

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}


// exact positional read into an arbitrary destination (no alignment games: used for F_NOCACHE and
// buffered reads that land directly in a host-visible cache slot)
static bool llama_moe_stream_pread_exact(int fd, uint8_t * dst, size_t len, size_t offs) {
    while (len > 0) {
        ssize_t r = pread(fd, dst, len, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (r == 0) {
            return false;
        }
        dst  += r;
        len  -= (size_t) r;
        offs += (size_t) r;
    }
    return true;
}


// direct read into a cache slot under F_NOCACHE: the uncached path wants 4 KiB-aligned offsets and
// lengths, so the aligned middle goes straight to the slot and the unaligned head and tail (< 4 KiB
// each) bounce through the staging buffer
static bool llama_moe_stream_pread_slot(int fd, uint8_t * dst, size_t len, size_t offs, uint8_t * staging, bool direct) {
    if (!direct) {
        return llama_moe_stream_pread_exact(fd, dst, len, offs);
    }
    const size_t a    = MOE_STREAM_DIRECT_ALIGN;
    const size_t head = (a - (offs % a)) % a;          // bytes to the first aligned offset
    if (head >= len) {
        return llama_moe_stream_pread_exact(fd, dst, len, offs);
    }
    const size_t mid  = ((len - head) / a) * a;         // aligned middle
    const size_t tail = len - head - mid;
    if (head > 0) {
        if (!llama_moe_stream_pread_exact(fd, staging, head, offs)) { return false; }
        memcpy(dst, staging, head);
    }
    if (mid > 0 && !llama_moe_stream_pread_exact(fd, dst + head, mid, offs + head)) {
        return false;
    }
    if (tail > 0) {
        if (!llama_moe_stream_pread_exact(fd, staging, tail, offs + head + mid)) { return false; }
        memcpy(dst + head + mid, staging, tail);
    }
    return true;
}


// read one contiguous file range into several destinations, split into `chunks` 4 KiB-aligned pieces
// read concurrently (GCD on Apple, sequential elsewhere)
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
static bool llama_moe_stream_pread_chunked(int fd, const std::vector<iovec> & iov, size_t offs, int chunks) {
    // flatten destinations into (file offset, dst, len) pieces, then cut into `chunks` aligned ranges
    struct piece { size_t off; uint8_t * dst; size_t len; };
    std::vector<piece> pieces;
    size_t total = 0;
    for (const auto & v : iov) { total += v.iov_len; }
    const size_t a = MOE_STREAM_DIRECT_ALIGN;
    const size_t step = ((total / chunks + a - 1) / a) * a;
    size_t cur_off = 0; size_t vi = 0; size_t v_done = 0;
    for (int c = 0; c < chunks && cur_off < total; c++) {
        size_t end = std::min(total, cur_off + step);
        if (c == chunks - 1) { end = total; }
        size_t pos = cur_off;
        while (pos < end) {
            while (vi < iov.size() && v_done == iov[vi].iov_len) { vi++; v_done = 0; }
            if (vi >= iov.size()) { break; }
            const size_t n = std::min(end - pos, iov[vi].iov_len - v_done);
            pieces.push_back({ offs + pos, (uint8_t *) iov[vi].iov_base + v_done, n });
            pos += n; v_done += n;
        }
        cur_off = end;
    }
    std::atomic<int> failed{0};
    std::atomic<int> * pfailed = &failed; // blocks capture by value; a pointer to the atomic is fine
    const piece * ppieces = pieces.data();
#ifdef __APPLE__
    dispatch_apply(pieces.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        if (!llama_moe_stream_pread_exact(fd, ppieces[i].dst, ppieces[i].len, ppieces[i].off)) {
            pfailed->store(1);
        }
    });
#else
    for (const auto & pc : pieces) {
        if (!llama_moe_stream_pread_exact(fd, pc.dst, pc.len, pc.off)) { pfailed->store(1); }
    }
#endif
    return failed.load() == 0;
}

// scattered positional read of one contiguous file range into several destinations
static bool llama_moe_stream_preadv_exact(int fd, std::vector<iovec> & iov, size_t offs) {
#ifdef _WIN32
    GGML_UNUSED(fd); GGML_UNUSED(iov); GGML_UNUSED(offs);
    return false;
#else
    size_t idx = 0;
    while (idx < iov.size()) {
        ssize_t r = preadv(fd, iov.data() + idx, (int) (iov.size() - idx), offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (r == 0) {
            return false;
        }
        offs += (size_t) r;
        size_t n = (size_t) r;
        while (idx < iov.size() && n >= iov[idx].iov_len) {
            n -= iov[idx].iov_len;
            idx++;
        }
        if (idx < iov.size() && n > 0) {
            iov[idx].iov_base = (uint8_t *) iov[idx].iov_base + n;
            iov[idx].iov_len -= n;
        }
    }
    return true;
#endif
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct, uint64_t ram_budget)
        : n_slots(n_slots), ram_budget(ram_budget) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;

    if (const char * e = std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD")) {
        lookahead = std::max(0, std::min(4, atoi(e)));
    }
    if (const char * e = std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD_K")) {
        // "6" applies to every distance, "8,4" per distance (the last value repeats)
        int last = 0;
        for (int d = 0; d < 4; d++) {
            if (e && *e) {
                last = std::max(0, atoi(e));
                const char * comma = strchr(e, ',');
                e = comma ? comma + 1 : nullptr;
            }
            lookahead_k[d] = last;
        }
    }
    lookahead_stat = std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD_STAT") != nullptr;
    if (lookahead > 0) {
        LLAMA_LOG_INFO("%s: MoE expert streaming lookahead prefetch: distance %d, ranks %s%s\n",
                __func__, lookahead, lookahead_k[0] > 0 ? std::to_string(lookahead_k[0]).c_str() : "all",
                lookahead_stat ? " (measure only)" : "");
    }
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    if (pack_fd >= 0) {
        close(pack_fd);
        pack_fd = -1;
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
        q_prefetch.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_claimed .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });
    sl->weights.back().src_name = meta->name;

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
#ifdef __APPLE__
            // macOS has no O_DIRECT; F_NOCACHE keeps streamed expert reads out of the unified buffer cache
            if (direct) {
                fcntl(files.back()->file_id(), F_NOCACHE, 1);
            }
#endif
        }
    };

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
#ifdef __APPLE__
        bool ok = !files.empty();
#else
        bool ok = !files.empty() && files.front()->has_direct_io();
#endif
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    detect_direct_write();

    {
        std::string pack = paths.empty() ? "" : paths.front() + ".epack";
        if (const char * e = getenv("LLAMA_MOE_STREAM_PACK")) {
            pack = e;
        }
        if (!pack.empty() && access(pack.c_str(), R_OK) == 0) {
            open_pack(pack);
        }
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}


// expert pack header, little-endian:
//   char magic[8] = "LMEPACK1", u32 version = 1, u32 n_layer, u32 n_expert, u32 blob_align, u64 header_size
//   per layer: u32 n_weights, u32 pad, u64 base_off, u64 blob_stride
//     per weight: char name[96], u32 ggml_type, u32 pad, u64 nb_expert, u64 slab_off
struct llama_moe_stream_pack_weight {
    char     name[96];
    uint32_t type;
    uint32_t pad;
    uint64_t nb_expert;
    uint64_t slab_off;
};

void llama_moe_stream::open_pack(const std::string & path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        LLAMA_LOG_WARN("%s: expert pack %s: open failed (%s), ignored\n", __func__, path.c_str(), strerror(errno));
        return;
    }
    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("%s: expert pack %s: %s, ignored\n", __func__, path.c_str(), why);
        close(fd);
        for (auto & sl : layers) {
            if (sl) { sl->packed = false; }
        }
    };

    struct { char magic[8]; uint32_t version, n_layer, n_expert, blob_align; uint64_t header_size; } hdr;
    if (!llama_moe_stream_pread_exact(fd, (uint8_t *) &hdr, sizeof(hdr), 0) ||
        memcmp(hdr.magic, "LMEPACK1", 8) != 0 || hdr.version != 1) {
        fail("bad header");
        return;
    }
    if (hdr.n_layer < layers.size()) {
        fail("layer count mismatch");
        return;
    }

    size_t pos = sizeof(hdr);
    uint32_t n_packed = 0;
    for (uint32_t il = 0; il < hdr.n_layer; il++) {
        struct { uint32_t n_weights, pad; uint64_t base_off, blob_stride; } le;
        if (!llama_moe_stream_pread_exact(fd, (uint8_t *) &le, sizeof(le), pos)) {
            fail("truncated layer table");
            return;
        }
        pos += sizeof(le);
        std::vector<llama_moe_stream_pack_weight> pw(le.n_weights);
        if (le.n_weights > 0 && !llama_moe_stream_pread_exact(fd, (uint8_t *) pw.data(), pw.size()*sizeof(pw[0]), pos)) {
            fail("truncated weight table");
            return;
        }
        pos += pw.size()*sizeof(pw[0]);

        if (il >= layers.size()) {
            continue; // the pack may describe layers this model does not stream (e.g. a NextN block)
        }
        auto & sl = layers[il];
        if (!sl || le.n_weights == 0) {
            continue;
        }
        if (sl->n_expert != hdr.n_expert || le.n_weights != sl->weights.size()) {
            continue; // layer not packed for this model shape
        }
        bool ok = true;
        for (auto & wt : sl->weights) {
            bool found = false;
            for (const auto & p : pw) {
                if (strncmp(p.name, wt.src_name.c_str(), sizeof(p.name)) == 0) {
                    found = p.type == (uint32_t) wt.cache->type && p.nb_expert == wt.nb_expert;
                    if (found) {
                        wt.pack_off = p.slab_off;
                    }
                    break;
                }
            }
            ok = ok && found;
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: expert pack: layer %u does not match the model tensors, read from GGUF\n", __func__, il);
            continue;
        }
        sl->packed      = true;
        sl->pack_base   = le.base_off;
        sl->pack_stride = le.blob_stride;
        size_t nb_blob = 0;
        for (const auto & wt : sl->weights) { nb_blob += wt.nb_expert; }
        max_nb_blob = std::max(max_nb_blob, nb_blob);
        n_packed++;
    }

#ifdef __APPLE__
    if (use_direct_io) {
        fcntl(fd, F_NOCACHE, 1);
    }
#endif
    pack_fd   = fd;
    pack_path = path;
    LLAMA_LOG_INFO("%s: expert pack %s: %u of %zu layers packed, one read per expert\n", __func__, path.c_str(), n_packed, layers.size());
}

// A cache slab can be filled through its host pointer when the buffer is a CPU buffer or a Metal
// shared buffer (unified memory). Verify per buffer by round-tripping a pattern through the
// backend's own set_tensor and the raw pointer, before any expert is loaded.
void llama_moe_stream::detect_direct_write() {
#ifdef _WIN32
    return;
#else
    if (getenv("LLAMA_MOE_STREAM_NO_DIRECT_WRITE")) {
        return;
    }
    // O_DIRECT needs block-aligned destinations, which arbitrary slot offsets are not; macOS uses
    // F_NOCACHE which has no such constraint
#ifndef __APPLE__
    if (use_direct_io) {
        return;
    }
#endif
    std::vector<ggml_backend_buffer_t> checked_ok, checked_bad;
    size_t n_direct = 0, n_total = 0;
    for (auto & sl : layers) {
        if (!sl) {
            continue;
        }
        for (auto & wt : sl->weights) {
            n_total++;
            ggml_backend_buffer_t buf = wt.cache->buffer;
            if (buf == nullptr || wt.cache->data == nullptr) {
                continue;
            }
            if (std::find(checked_bad.begin(), checked_bad.end(), buf) != checked_bad.end()) {
                continue;
            }
            if (std::find(checked_ok.begin(), checked_ok.end(), buf) == checked_ok.end()) {
                bool ok = ggml_backend_buffer_is_host(buf);
                if (!ok) {
                    const std::string name = ggml_backend_buffer_name(buf);
                    const bool metal_shared = name.rfind("MTL", 0) == 0 &&
                        name.find("Private") == std::string::npos && name.find("Mapped") == std::string::npos;
                    if (metal_shared) {
                        // round-trip: write via the backend, read via the pointer, then restore
                        const size_t n = std::min<size_t>(4096, wt.nb_expert);
                        std::vector<uint8_t> pat(n), back(n), zero(n, 0);
                        for (size_t i = 0; i < n; i++) { pat[i] = (uint8_t) (0xA5 ^ (i*7)); }
                        ggml_backend_tensor_set(wt.cache, pat.data(), 0, n);
                        memcpy(back.data(), wt.cache->data, n);
                        ok = back == pat;
                        ggml_backend_tensor_set(wt.cache, zero.data(), 0, n);
                    }
                }
                (ok ? checked_ok : checked_bad).push_back(buf);
                if (!ok) {
                    continue;
                }
            }
            wt.direct_write = true;
            n_direct++;
        }
    }
    LLAMA_LOG_INFO("%s: %zu of %zu streamed weights fill their cache slots by direct read\n", __func__, n_direct, n_total);
#endif
}

// runs n_jobs over the I/O thread pool, each thread with its own aligned staging buffer
static bool llama_moe_stream_run_jobs(int32_t n_threads, size_t n_jobs, size_t nb_staging,
        const std::function<bool(size_t, uint8_t *)> & job) {
    std::atomic<size_t> next(0);
    std::atomic<bool>   ok(true);

    std::vector<std::thread> pool;
    pool.reserve(n_threads);

    for (int32_t t = 0; t < n_threads; t++) {
        pool.emplace_back([&]() {
            uint8_t * staging = (uint8_t *) moe_aligned_alloc(nb_staging);
            if (staging == nullptr) {
                ok = false;
                return;
            }
            for (size_t i = next++; i < n_jobs && ok; i = next++) {
                if (!job(i, staging)) {
                    ok = false;
                }
            }
            moe_aligned_free(staging);
        });
    }
    for (auto & th : pool) {
        th.join();
    }

    return ok;
}

void llama_moe_stream::pin_partition(uint32_t n_expert_used) {
    // the dynamic pool must hold one wave plus its preload plus the parked pairs
    uint32_t n_dyn = std::max<uint32_t>(3*n_expert_used, 8);
    if (const char * e = getenv("LLAMA_MOE_STREAM_DYN")) { // benchmark knob: size of the dynamic (hotness) pool
        n_dyn = std::max<uint32_t>(n_dyn, (uint32_t) atoi(e));
    }

    const int64_t t_start = ggml_time_us();

    struct job { llama_moe_stream_weight * wt; uint32_t n; };
    std::vector<job> jobs;

    uint32_t n_pinned = 0;
    uint32_t n_expert = 0;

    for (auto & sl_ptr : layers) {
        if (sl_ptr == nullptr || sl_ptr->n_slots <= n_dyn) {
            continue;
        }
        auto & sl = *sl_ptr;

        sl.n_pinned   = sl.n_slots - n_dyn;
        n_pinned      = sl.n_pinned;
        this->n_pinned = sl.n_pinned;
        n_expert    = sl.n_expert;

        for (auto & wt : sl.weights) {
            jobs.push_back({ &wt, sl.n_pinned });
        }

        for (uint32_t e = 0; e < sl.n_pinned; e++) {
            sl.slot_expert[e]   = e;
            sl.slot_state[e]    = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            sl.slot_last_use[e] = ++sl.use_counter;
            sl.expert_slot[e]   = e;
            sl.seen[e]          = 1;
        }
    }

    const bool ok = llama_moe_stream_run_jobs(n_io_threads, jobs.size(), MOE_STREAM_FILL_CHUNK + 2*MOE_STREAM_DIRECT_ALIGN,
            [&](size_t i, uint8_t * staging) {
        const auto & j = jobs[i];
        const uint32_t g = std::max<uint32_t>(MOE_STREAM_FILL_CHUNK/j.wt->nb_expert, 1);

        for (uint32_t e = 0; e < j.n; e += g) {
            const uint32_t n   = std::min(g, j.n - e);
            const size_t   nb  = (size_t) n*j.wt->nb_expert;
            const size_t   off = (size_t) e*j.wt->nb_expert;

            const uint8_t * src = llama_moe_stream_pread(*files[j.wt->file_idx], staging, nb, j.wt->offs + off, use_direct_io);
            if (src == nullptr) {
                return false;
            }
            ggml_backend_tensor_set(j.wt->cache, src, off, nb);
        }
        return true;
    });

    if (!ok) {
        throw std::runtime_error("MoE expert streaming: failed to load the pinned expert partition");
    }

    LLAMA_LOG_INFO("%s: %u of %u experts resident per layer, %u dynamic slots (%.2f s)\n",
            __func__, n_pinned, n_expert, n_dyn, (ggml_time_us() - t_start)/1e6);
}

void llama_moe_stream::fill_host_mirror() {
    if (ram_budget == 0) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    struct job { llama_moe_stream_weight * wt; uint32_t first; uint32_t last; uint8_t * dst; };
    std::vector<job> jobs;

    size_t   left       = ram_budget;
    uint32_t n_layers   = 0;
    size_t   n_mirrored = 0;

    for (auto & sl_ptr : layers) {
        if (sl_ptr == nullptr) {
            continue;
        }
        auto & sl = *sl_ptr;

        const uint32_t n_host = sl.n_expert - sl.n_pinned;
        if (n_host == 0) {
            continue;
        }

        size_t nb_layer = 0;
        for (const auto & wt : sl.weights) {
            nb_layer += (size_t) n_host*wt.nb_expert;
        }
        if (nb_layer > left) {
            break;
        }

        // the mirror feeds device copies, so it lives in the backend host buffer to keep the DMA direct
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(sl.weights[0].cache->buffer)));
        ggml_backend_buffer_t buf = buft ? ggml_backend_buft_alloc_buffer(buft, nb_layer) : nullptr;
        if (buf == nullptr) {
            LLAMA_LOG_WARN("%s: host mirror allocation failed at layer %d, mirroring stops here\n", __func__, sl.il);
            break;
        }
        host_bufs.emplace_back(buf);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        for (auto & wt : sl.weights) {
            jobs.push_back({ &wt, sl.n_pinned, sl.n_expert, base });
            wt.host       = base;
            wt.host_first = sl.n_pinned;
            base         += (size_t) n_host*wt.nb_expert;
        }

        left       -= nb_layer;
        n_mirrored += nb_layer;
        n_layers++;
    }

    const bool ok = llama_moe_stream_run_jobs(n_io_threads, jobs.size(), MOE_STREAM_FILL_CHUNK + 2*MOE_STREAM_DIRECT_ALIGN,
            [&](size_t i, uint8_t * staging) {
        const auto & j = jobs[i];
        const uint32_t g = std::max<uint32_t>(MOE_STREAM_FILL_CHUNK/j.wt->nb_expert, 1);

        for (uint32_t e = j.first; e < j.last; e += g) {
            const uint32_t n   = std::min(g, j.last - e);
            const size_t   nb  = (size_t) n*j.wt->nb_expert;

            const uint8_t * src = llama_moe_stream_pread(*files[j.wt->file_idx], staging, nb, j.wt->offs + (size_t) e*j.wt->nb_expert, use_direct_io);
            if (src == nullptr) {
                return false;
            }
            memcpy(j.dst + (size_t) (e - j.first)*j.wt->nb_expert, src, nb);
        }
        return true;
    });

    if (!ok) {
        throw std::runtime_error("MoE expert streaming: failed to fill the expert host mirror");
    }

    uint32_t n_streamed = 0;
    for (const auto & sl_ptr : layers) {
        if (sl_ptr != nullptr && sl_ptr->n_expert > sl_ptr->n_pinned) {
            n_streamed++;
        }
    }

    LLAMA_LOG_INFO("%s: expert host mirror = %.2f GiB over %u layers (%.2f s)\n",
            __func__, n_mirrored / 1024.0 / 1024.0 / 1024.0, n_layers, (ggml_time_us() - t_start)/1e6);
    if (n_layers < n_streamed) {
        LLAMA_LOG_INFO("%s: %u of %u streamed layers are unmirrored, their misses read the model file\n",
                __func__, n_streamed - n_layers, n_streamed);
    }
}

void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: drains a batch of reserved loads, uploads their expert slab(s) into the cache
// slots, and marks the slots RESIDENT (or flags load_failed); stale/duplicate items are skipped.
// Mirror slabs stream async on a private per-device backend with a single stream sync per batch,
// so the copies pipeline back to back on the bus; file slabs go through the staging buffer and
// stay synchronous because the buffer is reused per slab
void llama_moe_stream::worker_loop() {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(std::max(max_nb_expert, max_nb_blob) + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    // async uploads need the buffer to sit in the device's default buffer type; anything else
    // (host or CPU placed caches) takes the synchronous path
    std::vector<std::pair<ggml_backend_dev_t, ggml_backend_t>> backends;
    auto backend_for = [&](ggml_backend_buffer_t buf) -> ggml_backend_t {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
        ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
        if (dev == nullptr || ggml_backend_dev_buffer_type(dev) != buft) {
            return nullptr;
        }
        for (const auto & b : backends) {
            if (b.first == dev) {
                return b.second;
            }
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        backends.emplace_back(dev, be);
        return be;
    };

    std::vector<llama_moe_stream_work> batch;
    std::vector<uint8_t>               batch_ok;

    // LLAMA_MOE_STREAM_CHUNKS=N (default 4 on Apple, 1 elsewhere): parallel sub-reads per packed miss
    int chunks = 1;
#ifdef __APPLE__
    chunks = 4;
#endif
    if (const char * e = getenv("LLAMA_MOE_STREAM_CHUNKS")) {
        chunks = std::max(1, atoi(e));
    }

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty() || !q_prefetch.empty(); });
        if (shutting_down) {
            break;
        }

        batch.clear();
        while ((!q_demand.empty() || !q_prefetch.empty()) && batch.size() < MOE_STREAM_UPLOAD_BATCH) {
            // demand loads first; a prefetch item is taken only when nothing is waiting on a read
            auto & q = !q_demand.empty() ? q_demand : q_prefetch;
            llama_moe_stream_work w = q.front();
            q.pop_front();
            if (&q == &q_prefetch && !batch.empty()) {
                // a prefetch never rides along in a demand batch (it would delay the demand commit)
                q.push_front(w);
                break;
            }

            auto & sl = *w.sl;
            if (w.gen != sl.slot_gen[w.slot] ||
                sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
                sl.slot_expert[w.slot] != w.expert ||
                sl.slot_claimed[w.slot]) {
                continue; // stale or duplicate item
            }
            sl.slot_claimed[w.slot] = 1;
            batch.push_back(w);
        }
        if (batch.empty()) {
            continue;
        }

        lk.unlock();

        batch_ok.assign(batch.size(), 1);
        std::vector<iovec> iov;
        const int64_t t_batch0 = ggml_time_us();
        int64_t t_handoff_acc = 0;
        for (size_t k = 0; k < batch.size(); k++) {
            t_handoff_acc += t_batch0 - batch[k].t_push_us;
        }
        for (size_t k = 0; k < batch.size(); k++) {
            const auto & w  = batch[k];
            const auto & sl = *w.sl;

            // packed expert: one contiguous blob, scattered into the slabs with a single preadv
            const bool mirrored = !sl.weights.empty() && sl.weights[0].host != nullptr && (uint32_t) w.expert >= sl.weights[0].host_first;
            if (sl.packed && !mirrored) {
                bool all_direct = true;
                size_t nb_total = 0;
                for (const auto & wt : sl.weights) {
                    all_direct = all_direct && wt.direct_write;
                    nb_total  += wt.nb_expert;
                }
                // slabs are laid out in pack_off order; preadv needs them in file order
                std::vector<const llama_moe_stream_weight *> order;
                for (const auto & wt : sl.weights) { order.push_back(&wt); }
                std::sort(order.begin(), order.end(), [](const llama_moe_stream_weight * a, const llama_moe_stream_weight * b) { return a->pack_off < b->pack_off; });

                const size_t blob_offs = sl.pack_base + (size_t) w.expert*sl.pack_stride;
                iov.clear();
                if (all_direct) {
                    size_t expect = 0;
                    bool contiguous = true;
                    for (const auto * wt : order) {
                        contiguous = contiguous && wt->pack_off == expect;
                        expect += wt->nb_expert;
                        iov.push_back({ (uint8_t *) wt->cache->data + (size_t) w.slot*wt->nb_expert, wt->nb_expert });
                    }
                    if (!contiguous) {
                        batch_ok[k] = 0;
                    } else if (chunks > 1 && batch.size() < (size_t) n_io_threads) {
                        // few misses in flight: split this blob into aligned chunks and read them in parallel,
                        // so one expert's latency is ~1/chunks instead of one thread's sequential read
                        if (!llama_moe_stream_pread_chunked(pack_fd, iov, blob_offs, chunks)) {
                            batch_ok[k] = 0;
                        }
                    } else if (!llama_moe_stream_preadv_exact(pack_fd, iov, blob_offs)) {
                        batch_ok[k] = 0;
                    }
                } else {
                    // staging holds the whole blob (max_nb_blob), then each slab goes through set_tensor
                    const uint8_t * data = llama_moe_stream_pread_exact(pack_fd, staging, nb_total, blob_offs) ? staging : nullptr;
                    if (data == nullptr) {
                        batch_ok[k] = 0;
                    } else {
                        for (const auto * wt : order) {
                            ggml_backend_tensor_set(wt->cache, data + wt->pack_off, (size_t) w.slot*wt->nb_expert, wt->nb_expert);
                        }
                    }
                }
                continue;
            }

            for (const auto & wt : sl.weights) {
                const size_t offs_slot = (size_t) w.slot*wt.nb_expert;

                if (wt.host != nullptr && (uint32_t) w.expert >= wt.host_first) {
                    const uint8_t * data = wt.host + (size_t) (w.expert - wt.host_first)*wt.nb_expert;

                    ggml_backend_t be = backend_for(wt.cache->buffer);
                    if (be != nullptr) {
                        ggml_backend_tensor_set_async(be, wt.cache, data, offs_slot, wt.nb_expert);
                    } else {
                        ggml_backend_tensor_set(wt.cache, data, offs_slot, wt.nb_expert);
                    }
                    continue;
                }

                // unified memory: read straight into the slot, no staging copy. Under F_NOCACHE an
                // unaligned slab costs three reads (head, aligned middle, tail), which measured slower
                // than one aligned read through staging, so uncached unpacked reads keep the staging path
                if (wt.direct_write && !use_direct_io) {
                    uint8_t * dst = (uint8_t *) wt.cache->data + offs_slot;
                    if (!llama_moe_stream_pread_slot(files[wt.file_idx]->file_id(), dst, wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, staging, use_direct_io)) {
                        batch_ok[k] = 0;
                        break;
                    }
                    continue;
                }

                const uint8_t * data = llama_moe_stream_pread(*files[wt.file_idx], staging, wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
                if (data == nullptr) {
                    batch_ok[k] = 0;
                    break;
                }
                ggml_backend_tensor_set(wt.cache, data, offs_slot, wt.nb_expert);
            }
        }

        for (const auto & b : backends) {
            if (b.second != nullptr) {
                ggml_backend_synchronize(b.second);
            }
        }
        const int64_t t_batch1 = ggml_time_us();

        lk.lock();
        stats.t_read_us    += t_batch1 - t_batch0;
        stats.n_read       += (int64_t) batch.size();
        stats.t_handoff_us += t_handoff_acc;

        for (size_t k = 0; k < batch.size(); k++) {
            const auto & w  = batch[k];
            auto &       sl = *w.sl;

            sl.slot_claimed[w.slot] = 0;
            if (!batch_ok[k]) {
                load_failed = true;
            } else {
                sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            }
        }
        cv_done.notify_all();
    }
    lk.unlock();

    for (const auto & b : backends) {
        if (b.second != nullptr) {
            ggml_backend_free(b.second);
        }
    }
    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = sl.n_pinned; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: worker reads = %" PRId64 ", time inside reads = %.2f ms (%.3f ms per load), remap->worker handoff = %.3f ms per load\n",
            __func__, stats.n_read, stats.t_read_us/1000.0, stats.n_read > 0 ? stats.t_read_us/1000.0/stats.n_read : 0.0, stats.n_read > 0 ? stats.t_handoff_us/1000.0/stats.n_read : 0.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    for (int d = 0; d < lookahead; d++) {
        if (stats.pred_n[d] == 0) {
            continue;
        }
        std::string ranks;
        for (int r = 0; r < 16 && stats.pred_rank_n[d][r] > 0; r++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%s%.0f", r ? "/" : "", 100.0*stats.pred_rank_hit[d][r]/stats.pred_rank_n[d][r]);
            ranks += buf;
        }
        LLAMA_LOG_INFO("%s: moe stream: lookahead d=%d: predictions = %" PRId64 ", confirmed = %.1f%% (by rank %s%%), "
                "demand misses covered = %" PRId64 "/%" PRId64 ", prefetches = %" PRId64 " (used = %" PRId64 ", %.1f%%)\n",
                __func__, d + 1, stats.pred_n[d], 100.0*stats.pred_hit[d]/stats.pred_n[d], ranks.c_str(),
                stats.pred_miss_covered[d], stats.pred_miss[d], stats.pred_fetch[d], stats.pred_fetch_used[d],
                stats.pred_fetch[d] > 0 ? 100.0*stats.pred_fetch_used[d]/stats.pred_fetch[d] : 0.0);
    }
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).

// GPU keep-warm pulse around the CPU-side remap (Metal, GGML_METAL_KEEP_WARM=pulse); resolved once
static void (*llama_moe_stream_warm_fn)(bool) = nullptr;
static void llama_moe_stream_warm_resolve() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL");
    if (reg == nullptr) {
        reg = ggml_backend_reg_by_name("Metal");
    }
    if (reg != nullptr) {
        llama_moe_stream_warm_fn = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_keep_warm_pulse");
    }
}
struct llama_moe_stream_warm_guard {
    llama_moe_stream_warm_guard()  { llama_moe_stream_warm_resolve(); if (llama_moe_stream_warm_fn) { llama_moe_stream_warm_fn(true);  } }
    ~llama_moe_stream_warm_guard() { if (llama_moe_stream_warm_fn) { llama_moe_stream_warm_fn(false); } }
};

// scores this layer's earlier predictions against the real routing (ids already in sl->uniq/touched,
// misses flagged in miss_of) and invalidates them
static void llama_moe_stream_score_predictions(llama_moe_stream_layer * sl, const std::vector<uint8_t> & is_miss) {
    auto * mgr = sl->mgr;
    for (size_t d = 0; d < sl->pred.size(); d++) {
        auto & pr = sl->pred[d];
        if (!pr.valid) {
            continue;
        }
        pr.valid = false;
        std::vector<uint8_t> predicted(sl->n_expert, 0);
        for (size_t i = 0; i < pr.ids.size(); i++) {
            const int32_t e = pr.ids[i];
            if (e < 0 || (uint32_t) e >= sl->n_expert) {
                continue;
            }
            const bool hit = sl->touched[e] != 0;
            predicted[e] = 1;
            mgr->stats.pred_n[d]++;
            mgr->stats.pred_hit[d] += hit;
            const int r = pr.n_per_token > 0 ? (int) (i % pr.n_per_token) : 0;
            if (r < 16) {
                mgr->stats.pred_rank_n[d][r]++;
                mgr->stats.pred_rank_hit[d][r] += hit;
            }
        }
        for (const int32_t e : sl->uniq) {
            if (is_miss[e]) {
                mgr->stats.pred_miss[d]++;
                mgr->stats.pred_miss_covered[d] += predicted[e];
            }
        }
        for (const int32_t e : pr.fetched) {
            mgr->stats.pred_fetch_used[d] += sl->touched[e] != 0;
        }
        pr.fetched.clear();
    }
}

// records layer L's prediction for layer L+d and prefetches the predicted experts that are not
// resident (lowest-priority loads into the coldest slots). called under mgr->mtx
static void llama_moe_stream_lookahead_locked(llama_moe_stream_layer * sl, int d,
        const int32_t * pred_ids, int64_t n_per_token, int64_t n_tokens) {
    auto * mgr = sl->mgr;
    llama_moe_stream_layer * nl = mgr->layer(sl->il + d);
    if (nl == nullptr) {
        return;
    }
    if ((size_t) d > nl->pred.size()) {
        nl->pred.resize(d);
    }
    auto & pr = nl->pred[d - 1];
    const int64_t n = n_per_token*n_tokens;
    pr.ids.assign(pred_ids, pred_ids + n);
    pr.n_per_token = (int32_t) n_per_token;
    pr.valid = true;
    pr.fetched.clear();

    if (mgr->lookahead_stat) {
        return;
    }

    // predicted experts are kept out of victim selection, so one prediction never evicts another
    std::vector<uint8_t> keep(nl->n_slots, 0);
    std::vector<int32_t> want;
    const int64_t k = mgr->lookahead_k[d - 1] > 0 ? std::min<int64_t>(mgr->lookahead_k[d - 1], n_per_token) : n_per_token;
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = pred_ids[i];
        if (e < 0 || (uint32_t) e >= nl->n_expert) {
            continue;
        }
        const auto it = nl->expert_slot.find(e);
        if (it != nl->expert_slot.end()) {
            keep[it->second] = 1;
            continue;
        }
        if (i % n_per_token < k && std::find(want.begin(), want.end(), e) == want.end()) {
            want.push_back(e);
        }
    }
    for (const int32_t e : want) {
        const int32_t v = mgr->pick_victim_locked(*nl, keep.data());
        if (v < 0) {
            break;
        }
        // the victim is the coldest resident expert; a wrong prediction costs one read and that slot.
        // (refusing victims hotter than the predicted expert measured no better)
        mgr->reserve_slot_locked(*nl, e, v);
        keep[v] = 1;
        mgr->q_prefetch.push_back({ nl, e, v, nl->slot_gen[v], ggml_time_us() });
        mgr->cv_work.notify_one();
        mgr->stats.pred_fetch[d - 1]++;
        pr.fetched.push_back(e);
    }
}

static void llama_moe_stream_remap_impl(llama_moe_stream_layer * sl, const ggml_tensor * a, ggml_tensor * dst,
        const ggml_tensor * pred_ids) {
    llama_moe_stream_warm_guard warm_guard;

    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    static thread_local std::vector<uint8_t> is_miss;
    is_miss.assign(sl->n_expert, 0);
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                // a load is queued or in flight (a prefetch, or a sibling token's miss): make sure it is served at demand priority
                mgr->q_demand.push_back({ sl, e, s, sl->slot_gen[s], ggml_time_us() });
                mgr->cv_work.notify_one();
                waited = true;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                mgr->cv_done.wait(lk);
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->q_demand.push_back({ sl, e, v, sl->slot_gen[v], ggml_time_us() });
            mgr->cv_work.notify_one();
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
            is_miss[e] = 1;
        }
    }

    // lookahead: score what earlier layers predicted for this one, then predict and prefetch for the
    // next layers while this layer's own misses load
    if (!sl->pred.empty()) {
        llama_moe_stream_score_predictions(sl, is_miss);
    }
    if (pred_ids != nullptr && mgr->lookahead > 0) {
        GGML_ASSERT(pred_ids->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(pred_ids));
        const int64_t n_per_token = pred_ids->ne[0];
        const int64_t n_tokens    = pred_ids->ne[1];
        const int64_t n_dist      = pred_ids->ne[2];
        for (int64_t d = 1; d <= n_dist && d <= mgr->lookahead; d++) {
            const int64_t off = (d - 1)*n_per_token*n_tokens;
            llama_moe_stream_lookahead_locked(sl, (int) d, (const int32_t *) pred_ids->data + off, n_per_token, n_tokens);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }
}

void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    llama_moe_stream_remap_impl((llama_moe_stream_layer *) userdata, a, dst, nullptr);
}

void llama_moe_stream_remap2(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    llama_moe_stream_remap_impl((llama_moe_stream_layer *) userdata, a, dst, b);
}

void llama_moe_stream_remap2_decode(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    llama_moe_stream_remap2(dst, a, b, ith, nth, userdata);
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (sl.touched[e]) {
            continue;
        }
        sl.touched[e] = 1;
        if ((uint32_t) e < sl.n_pinned) {
            sl.expert_wave[e] = 0; // permanently resident, rides in the first wave at no slot cost
        } else {
            sl.uniq.push_back(e);
        }
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = std::max<uint32_t>((uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity), 1);
    sl.plan_next_wave = 0;
}

// make wave w's expert slice (uniq[w*cap .. +count)) resident, waiting for its loads, and best-effort
// preload the next wave so its loads overlap this wave's compute. leaves sl.demand_slots = this wave's
// slots and sl.plan_pool = the resident parking pool (>= n_ids slots) the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    const size_t first = (size_t) w*sl.plan_capacity;
    const size_t count = first < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - first) : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        const size_t need = n_ids - count;
        if (sl.n_pinned >= need) {
            for (size_t i = 0; i < need; i++) {
                borrowed.push_back((int32_t) i);
            }
        } else {
            GGML_ASSERT(sl.plan_pool.size() >= need);
            for (size_t i = 0; i < need; i++) {
                borrowed.push_back(sl.plan_pool[i]);
                sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
            }
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    q_demand.push_back({ &sl, e, s, sl.slot_gen[s], ggml_time_us() }); // promote to demand, wait for it
                    cv_work.notify_one();
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                q_demand.push_back({ &sl, e, v, sl.slot_gen[v], ggml_time_us() });
                cv_work.notify_one();
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            q_demand.push_back({ &sl, e, v, sl.slot_gen[v], ggml_time_us() });
            cv_work.notify_one();
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_moe_stream_warm_guard warm_guard;
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}

void llama_moe_stream_remap_decode(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    llama_moe_stream_remap(dst, a, ith, nth, userdata);
}
