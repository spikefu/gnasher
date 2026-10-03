#!/bin/zsh
# Capture a Metal System Trace of a short GLM decode and aggregate GPU time per kernel.
# Needs full Xcode (xctrace). usage: gpu-trace.sh [label] [n_tokens]
set -e
if ! xcrun --find xctrace >/dev/null 2>&1; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer; fi
cd "$(dirname $0)"
label=${1:-glm}; ntok=${2:-32}
BIN=$HOME/Developer/gnasher/build/bin/llama-completion
MODEL=$HOME/models/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
out=results/trace-$label.trace
rm -rf "$out"
# record: the whole process, short run so the trace stays small
LLAMA_MOE_STREAM_DYN=200 LLAMA_ARG_MOE_STREAM_DIRECT=1 GGML_METAL_KEEP_WARM=pulse \
xcrun xctrace record --template 'Metal System Trace' --output "$out" --time-limit 90s \
    --launch -- "$BIN" -m "$MODEL" --moe-stream-cache 200s --moe-stream-ram 0 -lv 1 -n $ntok -c 4096 -no-cnv --temp 0 \
    -p "Explain, step by step, why a mixture-of-experts model needs a router. Be concise."
echo "--- tables in the trace:"
xcrun xctrace export --input "$out" --toc 2>/dev/null | grep -o 'schema="[^"]*"' | sort -u | tee results/trace-$label.toc.txt
echo "--- exporting GPU intervals (first matching schema):"
schema=$(grep -o -E 'schema="[^"]*(gpu|metal)[^"]*"' results/trace-$label.toc.txt | head -1 | cut -d'"' -f2)
echo "schema: $schema"
xcrun xctrace export --input "$out" --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" > results/trace-$label.xml 2>/dev/null
ls -la results/trace-$label.xml
python3 - results/trace-$label.xml <<'PY'
import sys, re, collections
xml=open(sys.argv[1]).read()
# generic aggregation: rows have <row> with child elements; collect (name/label, duration)
rows=re.findall(r'<row>(.*?)</row>', xml, re.S)
print("rows:", len(rows))
dur=collections.Counter(); cnt=collections.Counter()
for r in rows:
    name=re.search(r'<(?:kernel|shader|pipeline|label|name)[^>]*>([^<]*)<', r)
    d=re.search(r'<duration[^>]*>(\d+)<', r)
    if name and d:
        dur[name.group(1)]+=int(d.group(1)); cnt[name.group(1)]+=1
tot=sum(dur.values()) or 1
for k,v in dur.most_common(40):
    print(f"{100*v/tot:6.2f}% {v/1e6:9.2f} ms {cnt[k]:7d}x  {k}")
PY
