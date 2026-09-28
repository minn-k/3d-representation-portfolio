#!/usr/bin/env bash
# 한 에셋을 끝까지: (텍스트→이미지는 gen2d.py 로 미리) 이미지 → TRELLIS 3DGS → 학습 결과 형식 → APG 그래프 → 편집 데모
#   bash run_asset.sh <name> <image> "<edit_demo 옵션>"
set -euo pipefail
cd "$(dirname "$0")"
PY=/c/anaconda/anaconda3/envs/trellis/python.exe
name=$1; image=$2; edit_args=${3:-}
t0=$(date +%s)
$PY gen3d.py --image "$image" --name "$name" 2>&1 | grep "^\[gen3d\]"
t1=$(date +%s)
$PY to_model.py --name "$name"
$PY prepare_gen.py "$name" 2>&1 | grep -E "^\[gen\]|^\[prepare\]|^\[graph\]"
t2=$(date +%s)
$PY edit_demo.py --name "gen_$name" $edit_args 2>&1 | grep "^\[edit\]"
t3=$(date +%s)
echo "{\"gen3d_wall_s\": $((t1-t0)), \"graph_wall_s\": $((t2-t1)), \"edit_wall_s\": $((t3-t2))}" > "out/$name/wall.json"
cat "out/$name/wall.json"
