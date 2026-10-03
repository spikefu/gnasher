#!/bin/zsh
cd "$(dirname $0)"
BIN=$HOME/Developer/llama.cpp-moe-stream/build/bin/llama-server
MODEL=$HOME/models/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
python3 -c "import json;print(json.dumps({'prompt':open('prompt.txt').read(),'n_predict':256,'temperature':0,'cache_prompt':False}))" > /tmp/glm-req.json
for cfg in "srv-nomtp:" "srv-mtp4:--spec-type draft-mtp --spec-draft-n-max 4" "srv-mtp6:--spec-type draft-mtp --spec-draft-n-max 6"; do
  label=${cfg%%:*}; flags=${cfg#*:}
  LLAMA_MOE_STREAM_DYN=160 $BIN -m $MODEL --moe-stream-cache 160s --moe-stream-ram 0 -lv 4 -c 8192 --port 8089 ${=flags} > results/160s-$label.log 2>&1 &
  pid=$!
  until curl -s localhost:8089/health | grep -q '"ok"'; do sleep 2; if ! kill -0 $pid 2>/dev/null; then echo "== $label: server died"; grep -E "error|abort|Assert" results/160s-$label.log | head -5; continue 2; fi; done
  echo "== $label ($flags)"
  curl -s localhost:8089/completion -d @/tmp/glm-req.json -H 'Content-Type: application/json' | python3 -c "
import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print('content head:', r['content'][:90].replace(chr(10),' ')); print({k:round(v,2) for k,v in t.items() if k in ('prompt_per_second','predicted_per_second','predicted_n','predicted_ms','draft_n','draft_n_accepted')})"
  kill $pid; wait $pid 2>/dev/null
  grep -i -E "draft acceptance|acc per pos|hit rate" results/160s-$label.log | sed 's/^[0-9.]* [IWE] //' | tail -3
done
echo MTP-TEST-DONE
