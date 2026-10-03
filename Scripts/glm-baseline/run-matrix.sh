#!/bin/zsh
cd "$(dirname $0)"
while pgrep -f "llama-completion" >/dev/null; do sleep 5; done
./bench.sh 24s-buffered 24s ""
./bench.sh 48s-dyn-direct 48s 48 --moe-stream-direct
./bench.sh 48s-pinned-direct 48s "" --moe-stream-direct
./bench.sh 96s-dyn-direct 96s 96 --moe-stream-direct
./bench.sh 160s-dyn-direct 160s 160 --moe-stream-direct
echo MATRIX-DONE
