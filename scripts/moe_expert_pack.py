#!/usr/bin/env python3
"""Build an expert pack (.epack) for --moe-stream: one contiguous, page-aligned blob per
(layer, expert) holding all of that expert's routed slabs (gate, up, down or gate_up, down), so a
cache miss is a single read instead of one per tensor.

usage: moe_expert_pack.py <first-shard.gguf> [out.epack]   (default out: <first-shard>.epack)
"""
import glob, os, re, struct, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
import gguf  # noqa: E402

MAGIC = b"LMEPACK1"
ALIGN = 4096
EXPS = re.compile(r'^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$')
ORDER = {'gate_up': 0, 'gate': 0, 'up': 1, 'down': 2}

def shards(first):
    m = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', first)
    if not m:
        return [first]
    files = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))
    assert files, first
    return files

def main():
    first = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else first + ".epack"
    layers = {}  # il -> list of (kind, path, data_offset, nb_expert, type, name)
    n_layer_total = 0
    n_expert = None
    for path in shards(first):
        r = gguf.GGUFReader(path)
        arch = r.fields['general.architecture'].contents()
        bc = r.fields[f'{arch}.block_count'].contents()
        nextn = r.fields.get(f'{arch}.nextn_predict_layers')
        n_layer_total = max(n_layer_total, bc + (nextn.contents() if nextn else 0))
        for t in r.tensors:
            m = EXPS.match(t.name)
            if not m:
                continue
            il, kind = int(m.group(1)), m.group(2)
            ne = [int(x) for x in t.shape]
            # gguf-py reports shape in ggml order (ne0, ne1, ne2); experts are ne2
            n_exp = ne[2] if len(ne) >= 3 else 1
            if n_expert is None:
                n_expert = n_exp
            assert n_exp == n_expert, (t.name, ne)
            nb = int(t.n_bytes) // n_exp
            assert nb * n_exp == int(t.n_bytes)
            layers.setdefault(il, []).append((ORDER[kind], path, int(t.data_offset), nb, int(t.tensor_type), t.name))
    assert layers, "no routed expert tensors found"
    for il in layers:
        layers[il].sort()

    # header layout
    hdr_size = 8 + 4*4 + 8
    per_layer = 4 + 4 + 8 + 8
    per_weight = 96 + 4 + 4 + 8 + 8
    table_size = sum(per_layer + per_weight*len(layers.get(il, [])) for il in range(n_layer_total))
    data_start = (hdr_size + table_size + ALIGN - 1) // ALIGN * ALIGN

    # blob layout per layer
    plan = {}
    off = data_start
    for il in range(n_layer_total):
        ws = layers.get(il, [])
        if not ws:
            continue
        slab_offs, pos = [], 0
        for w in ws:
            slab_offs.append(pos); pos += w[3]
        stride = (pos + ALIGN - 1) // ALIGN * ALIGN
        plan[il] = (off, stride, slab_offs)
        off += stride * n_expert
    total = off
    print(f"{len(plan)} layers x {n_expert} experts, pack size {total/1e9:.1f} GB -> {out}")

    with open(out, 'wb') as f:
        f.write(MAGIC + struct.pack('<IIIIQ', 1, n_layer_total, n_expert, ALIGN, hdr_size + table_size))
        for il in range(n_layer_total):
            ws = layers.get(il, [])
            if il in plan:
                base, stride, slab_offs = plan[il]
                f.write(struct.pack('<IIQQ', len(ws), 0, base, stride))
                for w, so in zip(ws, slab_offs):
                    name = w[5].encode()[:95].ljust(96, b'\0')
                    f.write(name + struct.pack('<IIQQ', w[4], 0, w[3], so))
            else:
                f.write(struct.pack('<IIQQ', 0, 0, 0, 0))
        f.write(b'\0' * (data_start - f.tell()))

        fds = {}
        t0 = time.time(); done = 0
        for il in sorted(plan):
            base, stride, slab_offs = plan[il]
            ws = layers[il]
            for e in range(n_expert):
                blob = bytearray(stride)
                for w, so in zip(ws, slab_offs):
                    fd = fds.get(w[1]) or fds.setdefault(w[1], os.open(w[1], os.O_RDONLY))
                    chunk = os.pread(fd, w[3], w[2] + e*w[3])
                    assert len(chunk) == w[3], (w[5], e)
                    blob[so:so+w[3]] = chunk
                f.write(blob)
                done += stride
            el = time.time() - t0
            print(f"  layer {il:3d} done, {done/1e9:7.1f} GB, {done/1e9/max(el,1e-9):.2f} GB/s", flush=True)
        assert f.tell() == total, (f.tell(), total)
    print("ok")

if __name__ == '__main__':
    main()
