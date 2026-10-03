#!/bin/zsh
cd "$(dirname $0)"
BIN=$HOME/Developer/llama.cpp-moe-stream/build/bin
MODEL=$HOME/models/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
run() { # label slots extra...
  local label=$1 slots=$2; shift 2
  LLAMA_MOE_STREAM_DYN=$slots $BIN/llama-completion -m $MODEL --moe-stream-cache ${slots}s --moe-stream-ram 0 -lv 4 -c 8192 -n 256 -no-cnv --temp 0 -f prompt.txt "$@" > results/g1-$label.log 2>&1
  echo "== $label: $(grep -E '^.* eval time' results/g1-$label.log | grep -v prompt | sed -E 's/.*\(([^)]*)\).*/\1/') | $(grep -o 'hit rate = [0-9.]*%' results/g1-$label.log) | $(grep -E 'error|abort|failed to allocate|out of memory' results/g1-$label.log | head -1)"
}
echo "### I/O threads at 160 slots"
run t9  160 --moe-stream-io-threads 9
run t14 160 --moe-stream-io-threads 14
run t18 160 --moe-stream-io-threads 18
echo "### cache size"
run s176 176
run s200 200
echo "### parallel requests (server, 4 streams)"
python3 -c "
import json
p=open('prompt.txt').read()
for i in range(4): open(f'/tmp/glm-req{i}.json','w').write(json.dumps({'prompt':p+f'\n\n(Reply variant {i}.)','n_predict':128,'temperature':0,'cache_prompt':False}))"
LLAMA_MOE_STREAM_DYN=160 $BIN/llama-server -m $MODEL --moe-stream-cache 160s --moe-stream-ram 0 -lv 4 -c 16384 -np 4 --port 8089 > results/g1-par4.log 2>&1 & pid=$!
until curl -s localhost:8089/health | grep -q '"ok"'; do sleep 2; kill -0 $pid 2>/dev/null || break; done
t0=$(python3 -c 'import time;print(time.time())')
for i in 0 1 2 3; do curl -s localhost:8089/completion -d @/tmp/glm-req$i.json -H "Content-Type: application/json" > /tmp/glm-resp$i.json & pids+=($!); done; wait $pids
t1=$(python3 -c 'import time;print(time.time())')
python3 -c "
import json,sys
tot=0; rates=[]
for i in range(4):
    r=json.load(open(f'/tmp/glm-resp{i}.json')); t=r['timings']; tot+=t['predicted_n']; rates.append(round(t['predicted_per_second'],2))
wall=$t1-$t0
print(f'== par4: {tot} tokens in {wall:.1f}s wall = {tot/wall:.2f} tok/s aggregate; per-stream {rates}')"
kill $pid; wait $pid 2>/dev/null
echo GROUP1-DONE
