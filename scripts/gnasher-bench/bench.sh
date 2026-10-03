#!/bin/zsh
# GLM-5.3-Flash expert-streaming baseline on llama.cpp (rebased moe-stream branch).
# usage: bench.sh <label> <cache-spec e.g. 24s> [dyn-slots] [extra llama args...]
set -u
BIN=${BIN:-$HOME/Developer/llama.cpp-moe-stream/build/bin/llama-completion}
MODEL=${MODEL:-$HOME/models/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}
OUT=${OUT:-$HOME/Developer/gnasher/scripts/gnasher-bench/results}
label=$1; cache=$2; dyn=${3:-}; shift 3 2>/dev/null || shift $#
mkdir -p "$OUT"
log="$OUT/$label.log"
export LLAMA_MOE_STREAM_DYN=${dyn}
{
  echo "# label=$label cache=$cache dyn=$dyn extra=$*  date=$(date -Iseconds)"
  /usr/bin/time -l "$BIN" -m "$MODEL" --moe-stream-cache "$cache" --moe-stream-ram 0 -lv 4 \
    -c 8192 -n 256 -no-cnv --temp 0 -f "$(dirname $0)/prompt.txt" "$@"
} > "$log" 2>&1
echo "== $label"
grep -E -i "expert cache size|resident per layer|prompt eval time|^.* eval time|hit rate|load stall|wave stall|maximum resident|peak memory|error|abort" "$log" | sed 's/^[0-9.]* [IWE] //'
