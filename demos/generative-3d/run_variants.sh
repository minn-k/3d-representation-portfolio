#!/usr/bin/env bash
# 개수 줄인 변형: <n>_p150k (잘라내기만), <n>_d100k / <n>_d50k (distill.py 재최적화) → 그래프 → 편집 렌더
set -uo pipefail
cd "$(dirname "$0")"
PY=/c/anaconda/anaconda3/envs/trellis/python.exe
declare -A EDIT
EDIT[bear]="--grab-dir 1,0,1.3 --pull 1,-0.2,0.4 --dist 0.06 --pin-h 0.5 --grab-r 0.03 --eye 0.25,-0.7,0.38 --fovy 28 --focus 0.3"
EDIT[plant]="--grab-dir 1,-0.3,0.4 --pull 0.5,-0.2,1 --dist 0.07 --pin-h 0.45 --grab-r 0.025 --eye 0.3,-0.62,0.36 --fovy 24 --focus 0.5"
EDIT[robot]="--grab-dir 1,-0.2,0.35 --pull 0.3,-0.1,1 --dist 0.06 --pin-h 0.4 --pin-far 0.11 --grab-r 0.03 --eye 0.22,-0.72,0.36 --fovy 26 --focus 0.4"
for n in bear plant robot; do
  for var in p150k d100k d50k; do
    v=${n}_$var
    t1=$(date +%s)
    if [ $var = p150k ]; then $PY to_model.py --name $v --ply out/$n/gaussian.ply --keep 150000
    else $PY to_model.py --name $v --ply out/$n/distill_${var#d}.ply; fi
    $PY prepare_gen.py $v 2>&1 | grep -aE "components|SIBR"
    t2=$(date +%s)
    echo "{\"graph_wall_s\": $((t2-t1))}" > out/$v/wall.json
    $PY graph_stats.py $v > /dev/null
    $PY edit_demo.py --name gen_$v ${EDIT[$n]} 2>&1 | grep -a "^\[edit\] {" | sed 's/.*"xpbd_ms_per_step"/xpbd_ms/' | cut -c1-120
    echo "== $v done"
  done
done
