#!/bin/zsh
# 双击打开谱子：本地服务没开就在后台开起来（端口 8766），然后用默认浏览器打开总网页。
cd "$(dirname "$0")"
PY=/Users/s4146265/anaconda3/envs/guitarize/bin/python
SONG=${1:-nanmonee}
if ! curl -s -o /dev/null --max-time 2 http://localhost:8766/; then
  nohup "$PY" -m guitarize serve --port 8766 > /tmp/guitarize-serve.log 2>&1 &
  for i in {1..20}; do curl -s -o /dev/null --max-time 1 http://localhost:8766/ && break; sleep 0.5; done
fi
open "http://localhost:8766/songs/$SONG/out/index.html"
