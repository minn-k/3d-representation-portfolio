#!/usr/bin/env bash
# 글자 7개: TRELLIS → 5만 개 재최적화 → 학습 결과 형식 → APG 그래프
set -uo pipefail
cd "$(dirname "$0")"
PY=/c/anaconda/anaconda3/envs/trellis/python.exe
# 이름:입력 이미지 (T 는 이미 생성됨)
for spec in R:out/letter_R/i2i_s1.png E:out/letter_E/i2i_s3.png L:out/letter_L/i2i_s3.png L2:out/letter_L/i2i_s0.png I:out/letter_I/i2i_s0.png S:out/letter_S/i2i_s1.png; do
  L=${spec%%:*}; img=${spec#*:}
  $PY gen3d.py --image $img --name letter_$L 2>&1 | grep -a "^\[gen3d\]" | cut -c1-160
done
for L in T R E L L2 I S; do
  $PY distill.py --name letter_$L --keep 50000 --iters 3000 2>&1 | grep -a "^\[distill\]" | sed 's/.*"keep"/keep/'
  $PY to_model.py --name letter_${L}_d50k --ply out/letter_$L/distill_50k.ply
  $PY prepare_gen.py letter_${L}_d50k 2>&1 | grep -aE "components|SIBR"
  $PY graph_stats.py letter_${L}_d50k
done
