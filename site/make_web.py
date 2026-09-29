r"""종합 포트폴리오 웹페이지 (생성형 3D 데모 + APG-GS) 만들기.

  C:\anaconda\anaconda3\envs\trellis\python.exe make_web.py

만드는 것: index.html, assets/* (영상·그림 복사, 새 영상은 H.264 faststart 로 다시 인코딩)
숫자는 전부 genai/out/<name>/*.json (측정 로그) 에서 읽는다 — 손으로 쓰지 않는다.
기존 APG-GS 영상은 ../portfolio_site/assets 에서 가져온다.
"""
import html
import json
import os
import shutil
import subprocess

import numpy as np

import imageio_ffmpeg
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", r"C:\gaussian-splatting")      # genai/out · portfolio_site/assets · 덱 그림이 있는 작업 폴더
GEN = os.path.join(ROOT, "genai", "out")
OLD = os.path.join(ROOT, "portfolio_site", "assets")
MEDIA = os.path.join(ROOT, "portfolio_ppt_final", "build_v3", "media")
A = os.path.join(HERE, "assets")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

GITHUB = "https://github.com/minn-k/3d-representation-portfolio"
CODE = GITHUB + "/tree/main/demos/generative-3d"
EDIT_VAR = "d100k"                       # 편집 영상 · 편집 수치에 쓰는 판 (distill.py 10만 개)
VARIANTS = ("p150k", "d100k", "d50k")

# 생성 데모 에셋: (이름, 고른 시드, 한 줄 설명, 편집 설명)
ASSETS = [
    ("bear", 3, "곰 인형", "귀를 잡아 옆으로 당김 — 부드러운 물체"),
    ("plant", 0, "화분", "잎 끝을 잡아 들어 올림 — 얇은 구조"),
    ("robot", 3, "안내 로봇 캐릭터", "팔을 잡아 들어 올림 — 캐릭터 자세 편집"),
]


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def enc(src, dst, scale_w=None, crop_top=0, crop_bottom=0):
    """H.264 yuv420p + faststart (웹 재생용). crop_top / crop_bottom: 위 · 아래를 잘라낼 픽셀 (영상 안 글자)."""
    cut = crop_top + crop_bottom
    f = ([f"crop=iw:ih-{cut}:0:{crop_top}"] if cut else []) + ([f"scale={scale_w}:-2"] if scale_w else [])
    vf = ["-vf", ",".join(f)] if f else []
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, *vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-preset", "slow", "-movflags", "+faststart", "-an", dst], check=True)


def poster(src_img, dst, w=None):
    im = Image.open(src_img).convert("RGB")
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    im.save(dst, quality=88)


def split_grid(grid, base_dst, keep=5, legend=30):
    """부위 비교 그림 (--baselines 면 기준선 행이 더 붙음) → 앞 keep 행 + 범례 / 나머지 (기준선) 행.
    두 에셋의 최종 분류표를 같은 행으로 맞추고, 기준선은 따로 보여 준다."""
    im = Image.open(grid).convert("RGB")
    row = round(256 * im.width / 1024)
    n = (im.height - legend) // row
    if n <= keep:
        return
    leg = im.crop((0, im.height - legend, im.width, im.height))

    def stack(a, b, with_legend):
        out = Image.new("RGB", (im.width, (b - a) * row + (legend if with_legend else 0)), (255, 255, 255))
        out.paste(im.crop((0, a * row, im.width, b * row)), (0, 0))
        if with_legend:
            out.paste(leg, (0, (b - a) * row))
        return out
    stack(keep, n, False).save(base_dst, quality=88)                # 기준선 색은 부위가 아니라 묶음 번호 → 범례 없음
    stack(0, keep, True).save(grid, quality=88)


def first_frame(video, dst, t=0.0):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-q:v", "3", dst],
                   check=True)


def build_media():
    os.makedirs(A, exist_ok=True)
    rows = []
    for name, seed, title, edit in ASSETS:
        d = os.path.join(GEN, name)
        if not os.path.exists(os.path.join(d, "edit_stats.json")):
            print(f"skip {name}: not finished")
            continue
        poster(os.path.join(d, f"text2img_s{seed}.png"), os.path.join(A, f"gen_{name}_image.jpg"), 512)
        # TRELLIS 턴테이블은 뒷면에서 시작한다 → 앞면(1.7 s)부터 돌게 잘라 붙이고 포스터도 앞면
        tt = os.path.join(d, "turntable_front.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", os.path.join(d, "turntable.mp4"), "-filter_complex",
                        "[0]trim=start=1.7,setpts=PTS-STARTPTS[a];[0]trim=end=1.7,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1",
                        tt], check=True)
        enc(tt, os.path.join(A, f"gen_{name}_turntable.mp4"))
        first_frame(os.path.join(A, f"gen_{name}_turntable.mp4"), os.path.join(A, f"gen_{name}_turntable.jpg"))
        dv = os.path.join(GEN, f"{name}_{EDIT_VAR}")                   # 편집은 개수 줄인(재최적화) 판으로
        enc(os.path.join(dv, "edit_compare_only.mp4"), os.path.join(A, f"gen_{name}_edit.mp4"))   # 변형만 (Σ′ 적용)
        first_frame(os.path.join(A, f"gen_{name}_edit.mp4"), os.path.join(A, f"gen_{name}_edit.jpg"))
        t2i = jl(os.path.join(d, "text2img.json"))
        rows.append({
            "name": name, "title": title, "edit": edit,
            "prompt": t2i["prompt"].split(",")[0],
            "t2i": next(i for i in t2i["images"] if i["seed"] == seed),
            "g3": jl(os.path.join(d, "stats.json")),
            "tm": jl(os.path.join(d, "to_model.json")),
            "ed_full": jl(os.path.join(d, "edit_stats.json")),
            "wall_full": jl(os.path.join(d, "wall.json")),
            "ed": jl(os.path.join(dv, "edit_stats.json")),
            "graph": jl(os.path.join(dv, "graph.json")),
            "wall": jl(os.path.join(dv, "wall.json")),
            "var": {k: {"ed": jl(os.path.join(GEN, f"{name}_{k}", "edit_stats.json")),
                        "wall": jl(os.path.join(GEN, f"{name}_{k}", "wall.json"))} for k in VARIANTS},
            "distill": {k: jl(os.path.join(d, f"distill_{k}.json")) for k in ("100k", "50k")},
        })
    poster(os.path.join(GEN, "distill_grid.png"), os.path.join(A, "distill_grid.jpg"), 1200)
    ld = os.path.join(GEN, "letters", "letters_drop.mp4")
    if os.path.exists(ld):                                            # 맨 앞 영상: 글자 낙하 (letters_drop.py)
        enc(ld, os.path.join(A, "letters_drop.mp4"))
        first_frame(os.path.join(A, "letters_drop.mp4"), os.path.join(A, "letters_drop.jpg"), 4.0)
    # 기존 연구 영상·그림
    for f in ("prior_research_demo",):
        for ext in ("mp4", "jpg"):
            s = os.path.join(OLD, f"{f}.{ext}")
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(A, f"{f}.{ext}"))
    for f in ("cov_pull_rest", "cov_pull_posonly", "cov_pull_posshape"):
        poster(os.path.join(MEDIA, f"{f}.png"), os.path.join(A, f"{f}.jpg"), 720)
    # 의미 신호 → 부위 → 물리 (gen3d_sem.py · seg2d.py · lift_parts.py · edit_demo.py --parts)
    for n in ("robot_sem", "bear_sem"):
        d = os.path.join(GEN, n)
        if not os.path.exists(os.path.join(d, "parts3d.npz")):
            continue
        enc(os.path.join(d, "parts_turntable.mp4"), os.path.join(A, f"{n}_parts.mp4"))
        first_frame(os.path.join(A, f"{n}_parts.mp4"), os.path.join(A, f"{n}_parts.jpg"))
        poster(os.path.join(d, "parts2d.png"), os.path.join(A, f"{n}_parts2d.jpg"), 520)
        poster(os.path.join(d, "parts3d_grid.png"), os.path.join(A, f"{n}_parts3d_grid.jpg"), 1024)
        split_grid(os.path.join(A, f"{n}_parts3d_grid.jpg"), os.path.join(A, f"{n}_baselines.jpg"))
        if os.path.exists(os.path.join(d, "parts3d_vox_feat_diff.png")):
            poster(os.path.join(d, "parts3d_vox_feat_diff.png"), os.path.join(A, f"{n}_vox_feat_diff.jpg"), 1400)
        if os.path.exists(os.path.join(d, "camera_fit.png")):                 # lift_parts.py --mode proj
            poster(os.path.join(d, "camera_fit.png"), os.path.join(A, f"{n}_camera_fit.jpg"), 1400)
    pv = os.path.join(GEN, "part_shake", "robot_part2_stiff.mp4")
    if os.path.exists(pv):                                            # 부위를 아는 솔버 흔들기 (part_shake.py)
        enc(pv, os.path.join(A, "robot_part_shake.mp4"), crop_top=90, crop_bottom=100)  # 영상 안 글자는 캡션이 대신한다
        first_frame(os.path.join(A, "robot_part_shake.mp4"), os.path.join(A, "robot_part_shake.jpg"), 1.0)
    return rows


E = html.escape


def video(src, poster_, label="", auto=False):
    cap = f'<figcaption>{label}</figcaption>' if label else ""
    ap = " autoplay" if auto else ""
    return (f'<figure class="media"><video src="assets/{src}" poster="assets/{poster_}" controls muted loop playsinline{ap} '
            f'preload="metadata"></video>{cap}</figure>')


def img(src, label="", alt=""):
    cap = f'<figcaption>{label}</figcaption>' if label else ""
    return f'<figure class="media"><img src="assets/{src}" alt="{E(alt or label)}" loading="lazy">{cap}</figure>'


def fmt(n):
    return f"{n:,}"


def pct(x):
    """비율 → % 문자열. 아주 작은 값을 0.00% 로 뭉개지 않는다."""
    v = 100.0 * x
    return "0%" if v == 0 else (f"{v:.3f}%" if v < 0.01 else f"{v:.2f}%")


def asset_cards(rows):
    out = []
    for r in rows:
        out.append(f"""
      <article class="asset" id="gen-{r['name']}">
        <header><h4>{E(r['title'])}</h4><code class="prompt">“{E(r['prompt'])}”</code></header>
        <div class="asset-grid">
          {img(f"gen_{r['name']}_image.jpg", "① 텍스트 → 이미지 (SDXL-Turbo)")}
          {video(f"gen_{r['name']}_turntable.mp4", f"gen_{r['name']}_turntable.jpg", "② 이미지 → 3D Gaussians (TRELLIS)", True)}
          {video(f"gen_{r['name']}_edit.mp4", f"gen_{r['name']}_edit.jpg",
                 "③ 10만 개로 재최적화 → 잡아당겨 변형 (Σ′ = FΣ₀Fᵀ 적용)", True)}
        </div>
      </article>""")
    return "\n".join(out)


def metrics_table(rows):
    """생성 단계 (TRELLIS 원본)."""
    head = ("<tr><th>에셋</th><th>3D 생성<br><small>TRELLIS 샘플링</small></th><th>최대 VRAM<br><small>3D 생성</small></th>"
            "<th>생성된 가우시안</th><th>재최적화<br><small>→ 10만 개 · 3000회</small></th>"
            "<th>2배 넘게 늘어난 간선<br><small>10만 개 판 · 당긴 뒤 · 낮을수록 좋음</small></th></tr>")
    body = []
    for r in rows:
        g3, ed = r["g3"], r["ed"]
        body.append(f"<tr><td>{E(r['title'])}</td><td>{g3['generate_s']:.0f} s</td><td>{g3['peak_vram_gb']:.1f} GB</td>"
                    f"<td>{fmt(g3['gaussians'])}</td><td>{r['distill']['100k']['seconds']:.0f} s</td>"
                    f"<td>{pct(ed['edges_over_2x'])}</td></tr>")
    return f'<div class="table-wrap"><table class="metrics">{head}{"".join(body)}</table></div>'


def reduce_table(rows):
    """개수 조절: 화질(물체 픽셀 PSNR, 원본 렌더 대비) · 그래프 준비 · XPBD 1스텝."""
    head = ("<tr><th rowspan=2>에셋</th><th colspan=2>10만 개 · 물체 PSNR<br><small>원본 대비 · 높을수록 같음</small></th>"
            "<th colspan=2>5만 개 · 물체 PSNR</th>"
            "<th colspan=3>XPBD 1스텝 (20회 반복)<br><small>낮을수록 빠름</small></th>"
            "<th colspan=2>그래프 준비<br><small>변환부터 그래프까지</small></th>"
            "<th colspan=2>위치만 vs Σ′ 갱신<br><small>평균 픽셀 차이 (0~255)</small></th></tr>"
            "<tr><th>잘라내기만</th><th>재최적화</th><th>잘라내기만</th><th>재최적화</th>"
            "<th>원본</th><th>10만</th><th>5만</th><th>원본</th><th>10만</th><th>원본</th><th>10만</th></tr>")
    body = []
    for r in rows:
        d1, d5, v = r["distill"]["100k"], r["distill"]["50k"], r["var"]
        body.append(
            f"<tr><td>{E(r['title'])}<br><small>정리 후 {fmt(r['ed_full']['gaussians'])}개</small></td>"
            f"<td>{d1['psnr_obj_prune_only']:.1f} dB</td><td><b>{d1['psnr_obj_distilled']:.1f} dB</b></td>"
            f"<td>{d5['psnr_obj_prune_only']:.1f} dB</td><td><b>{d5['psnr_obj_distilled']:.1f} dB</b></td>"
            f"<td>{r['ed_full']['xpbd_ms_per_step']:.0f} ms</td><td><b>{v['d100k']['ed']['xpbd_ms_per_step']:.0f} ms</b></td>"
            f"<td><b>{v['d50k']['ed']['xpbd_ms_per_step']:.0f} ms</b></td>"
            f"<td>{r['wall_full']['graph_wall_s']} s</td><td><b>{v['d100k']['wall']['graph_wall_s']} s</b></td>"
            f"<td>{r['ed_full']['mean_abs_pixel_diff_pos_vs_shape']:.2f}</td>"
            f"<td><b>{v['d100k']['ed']['mean_abs_pixel_diff_pos_vs_shape']:.2f}</b></td></tr>")
    return f'<div class="table-wrap"><table class="metrics">{head}{"".join(body)}</table></div>'


def sem_section():
    ps_path = os.path.join(GEN, "part_shake", "robot_part2_stiff.json")
    pn_path = os.path.join(GEN, "part_shake", "robot_part_noshape.json")
    part_shake_html = ""
    if os.path.exists(ps_path) and os.path.exists(pn_path):          # 부위를 아는 솔버 (part_shake.py)
        ps, pn = jl(ps_path), jl(pn_path)                             # pn: 형상 유지 하나 (--no-part-shape, 몸 0.6 · 0.15)
        pu, pp, nn = ps["wobble_rms_cm"]["uniform"], ps["wobble_rms_cm"]["part"], pn["wobble_rms_cm"]["part"]
        bu, bp, bn = (100 * x["boundary"][m]["over_1p5_max"] for x, m in ((ps, "uniform"), (ps, "part"), (pn, "part")))
        ru, rp = ps["soft_rel_body_cm"]["uniform"]["rms"], ps["soft_rel_body_cm"]["part"]["rms"]
        part_shake_html = f"""
    <h3>부위마다 다른 물성 — 흔들기 비교</h3>
    <p class="sub">part_id 를 솔버에 넘기면 물성을 부위마다 다르게 줄 수 있다 — 여기서는 몸은 단단하게, 팔은 무르게.
      두 로봇 모두 <b>발만 받침에 고정한 한 덩어리 연체</b>이고 모양 · 그래프 · 솔버가 같다. 오른쪽만 part_id 를 넘겨
      부피 클러스터를 부위 안에서만 만들고, 형상 유지를 몸(머리 · 몸통 · 다리)과 두 팔로 나누고, 간선 강성을 경계에서 몸 → 팔로 매끄럽게 낮췄다.
      받침을 좌우 ±{ps['amp'] * 100:.0f} cm · {ps['freq']:.0f} Hz 로 {ps['shake_s']} s 흔든 뒤 멈춘다.</p>
    <figure class="media"><video src="assets/robot_part_shake.mp4" poster="assets/robot_part_shake.jpg" controls muted loop playsinline autoplay preload="metadata"></video><figcaption>왼쪽 기존 그래프 · 물성 하나 / 오른쪽 part_id 를 넘긴 솔버 (간선 강성 몸 {ps['body_stiff']} → 팔 {ps['soft_stiff']} · 형상 유지 몸 {ps['body_shape']} · 팔 {ps['soft_shape']})</figcaption></figure>
    <div class="table-wrap"><table class="metrics">
      <tr><th>발만 고정한 로봇<br><small>흔들림 RMS (cm) · 받침 이동을 뺀 값<br>팔은 몸의 움직임까지 뺀 값</small></th><th>머리</th><th>몸통</th><th>팔 (몸 기준)</th><th>경계 간선<br><small>1.5배 넘게 늘어난 비율<br>낮을수록 이음매가 자연스러움</small></th></tr>
      <tr><td>기존 그래프 · 물성 하나</td><td>{pu['head']:.2f}</td><td>{pu['torso']:.2f}</td><td>{ru:.2f}</td><td>{bu:.1f}%</td></tr>
      <tr><td>부피 · 형상 유지 · 강성 모두 부위별 (위 영상)</td><td>{pp['head']:.2f}</td><td>{pp['torso']:.2f}</td><td>{rp:.2f}</td><td>{bp:.1f}%</td></tr>
      <tr><td>부피 · 강성만 부위별, 형상 유지는 하나<br><small>몸 강성 {pn['body_stiff']} · 형상 유지 {pn['body_shape']}</small></td><td>{nn['head']:.2f}</td><td><b>{nn['torso']:.2f}</b></td><td><small>미측정</small></td><td><b>{bn:.1f}%</b></td></tr>
    </table></div>
    <p class="note">형상 유지를 두 팔에 따로 주면 두 기준이 만나는 어깨 이음매가 늘어난다 (경계 간선 {bp:.0f}%).
      형상 유지를 하나로 두고 부피 · 강성만 부위별로 하면 몸통 흔들림이 {pu['torso']:.2f} → {nn['torso']:.2f} cm, 경계 늘어남이 {bu:.1f} → {bn:.1f}% 로 줄었다.
      팔은 아직 더 흔들리지 않는다: 좌우로 흔들면 힘이 옆으로 뻗은 팔의 길이 방향으로 들어가 팔이 휘지 않고, 보이는 움직임은 대부분 머리가 끄덕이는 것이다.
      어느 부위를 단단 / 무르게 할지는 사람이 정했다.</p>"""

    rpart = np.load(os.path.join(GEN, "robot_sem", "parts3d.npz"))["asset_part"]
    rst = jl(os.path.join(GEN, "robot_sem", "parts3d_stats.json"))
    bst = jl(os.path.join(GEN, "bear_sem", "parts3d_stats.json"))
    bfeat = bst.get("vox_feat", {})
    bvoxels = bfeat.get("decoder", {}).get("voxels", 0) or bst.get("voxels", 0)
    return f"""
<section id="semantic">
  <div class="wrap">
    <p class="eyebrow">NEW · 생성 모델 내부 신호 → 가우시안 부위 → 부위별 물성</p>
    <h2>가우시안마다 '어느 부위인지' — 생성 모델 안에서 꺼내기</h2>
    <p class="sub">기하 그래프는 가우시안이 머리인지 팔인지 모른다. TRELLIS 는 3D 를 만들 때 <b>입력 이미지의 어느 부분을 보는지(cross-attention)</b>를 거치는데,
      이 신호로 사진의 2D 부위 이름을 3D 로 옮겼다. 사진에 보이는 쪽은 입력 카메라를 추정해 그대로 투영하고, 가려진 쪽은 부피를 따라 채운 뒤,
      경계는 복셀 특징 유사도로 다듬었다. 그 결과 <b>모든 가우시안이 part_id 와 신뢰도를 가진다</b> (로봇 {len(rpart):,}개 전부 · PLY 속성으로 내보냄).</p>
    <ol class="pipe">
      <li><span class="tag gen">2D</span><b>사진의 부위 이름</b><small>Grounding DINO + SAM</small></li>
      <li><span class="tag mine">보이는 쪽</span><b>2D 부위 투영</b><small>attention · 실루엣으로 입력 카메라 추정</small></li>
      <li><span class="tag mine">가려진 쪽</span><b>부피 기준</b><small>어느 부위의 속에 붙어 있나</small></li>
      <li><span class="tag mine">경계</span><b>복셀 특징 유사도</b><small>보이는 부위의 대표 특징과 비교</small></li>
      <li><span class="tag mine">결과</span><b>part_id → 부위별 물성</b><small>가우시안 = 자기가 나온 복셀의 라벨</small></li>
    </ol>
    <div class="row2">
      <figure class="media"><video src="assets/robot_sem_parts.mp4" poster="assets/robot_sem_parts.jpg" controls muted loop playsinline autoplay preload="metadata"></video><figcaption>안내 로봇 — 왼쪽 원래 색 · 오른쪽 가우시안마다 붙은 부위</figcaption></figure>
      <figure class="media"><video src="assets/bear_sem_parts.mp4" poster="assets/bear_sem_parts.jpg" controls muted loop playsinline autoplay preload="metadata"></video><figcaption>곰 인형 — 같은 방법 (아래 4차)</figcaption></figure>
    </div>
{part_shake_html}

    <h3>분류는 어떻게 다듬었나 — 곰 인형에서 4단계</h3>
    <p class="sub">좌표나 특징만으로 묶으면 이름 있는 부위가 나오지 않는다 (아래 기준선). 그래서 생성 모델이 사진의 어디를 보는지(attention)에서 출발했고,
      털 질감이 고른 곰 인형에서 부위가 섞이는 문제를 네 단계로 고쳤다. 단계별 그림은 같은 곰을 정면 · 옆 · 뒤 · 반대 옆에서 본 것.</p>
    <figure class="media trial"><img src="assets/bear_sem_baselines.jpg" alt="기준선 — 이름 없이 묶기" loading="lazy">
      <figcaption><b>기준선 · 이름 없이 묶기</b> — 위에서부터 DiT 특징 · decoder 특징 · 좌표 k-means.
        특징으로 묶으면 무늬대로, 좌표로 묶으면 위치대로 나뉘어 부위가 되지 않는다 (색 = 묶음 번호).</figcaption></figure>
    <figure class="media trial"><img src="assets/bear_parts_try1.jpg" alt="1차 — attention 투표 + DiT 특징 전파" loading="lazy">
      <figcaption><b>1차 · attention 투표 + DiT 특징 전파</b> — 오른팔 안쪽에 머리 라벨이 띠처럼 섞이고 다리에 머리 점이 생겼다.
        attention 은 대응점이 아니라 비슷한 특징을 찾는 것이라 고른 털에서는 엉뚱한 패치까지 보고, 가려진 복셀도 앞 픽셀로 투표했으며,
        넓은 전파 반경이 팔과 몸통 사이 틈을 건넜다.</figcaption></figure>
    <figure class="media trial"><img src="assets/bear_parts_try2.jpg" alt="2차 — 입력 카메라 추정 + 보이는 복셀만 투영" loading="lazy">
      <figcaption><b>2차 · 입력 카메라 추정 + 보이는 복셀만 투영</b> — 앞면은 깨끗해졌지만, 가려진 옆 · 뒤를 표면을 따라 번지는 전파로 채워
        팔이 몸통 옆과 허벅지까지 먹고 꼬리는 다리가 됐다 (사진에서 몸통 라벨은 배뿐이었다).</figcaption></figure>
    <figure class="media trial"><img src="assets/bear_sem_camera_fit.jpg" alt="2차의 입력 카메라 추정" loading="lazy">
      <figcaption>2차의 입력 카메라 추정 — attention 으로 초기값을 잡고 실루엣으로 다듬었다 (실루엣 일치 0.30 → 0.89).
        왼쪽 사진의 2D 부위 · 가운데 추정 카메라로 본 3D 부위 · 오른쪽 실루엣 비교 (회색 = 일치 · 빨강 = 복셀만 · 파랑 = 사진만).</figcaption></figure>
    <figure class="media trial"><img src="assets/bear_parts_try3.jpg" alt="3차 — 가려진 쪽은 부피 기준" loading="lazy">
      <figcaption><b>3차 · 가려진 쪽은 '어느 부위의 속(부피)에 붙어 있나' 로</b> — 복셀 껍질을 속이 찬 부피로 채우고, 가려진 표면마다
        부피 안에서 가장 가깝게 닿는 부위의 속을 따랐다 (두꺼운 속은 지나기 쉽고, 목처럼 얇은 연결은 어렵게).
        등과 꼬리가 배와 같은 몸통으로 돌아왔다. 뒷머리 · 목 뒤에는 팔 라벨이 아직 조금 섞인다.</figcaption></figure>
    <figure class="media trial"><img src="assets/bear_sem_vox_feat_diff.jpg" alt="4차 — 복셀 특징 유사도로 바뀐 위치" loading="lazy">
      <figcaption><b>4차 · 복셀 특징 유사도로 경계 보정</b> — Gaussian decoder 의 복셀 특징(768 → 64차원)으로, 사진에서 이름을 받은 복셀들의
        부위별 대표 특징과 닮은 정도를 약하게(20%) 투표하고 맞닿은 복셀끼리 전파했다. 사진에서 받은 라벨은 고정한다.
        곰에서 {bfeat.get('voxels_changed', 0):,} / {bvoxels:,} 복셀이 바뀌었고 (그림에서 색이 칠해진 곳), 뒷머리의 빨간 줄은 3차에서 팔로 섞였던 곳이 머리로 돌아온 것이다.</figcaption></figure>
    <p class="note trial-note">실제 곰에는 정답 라벨이 없어, 부위를 아는 <b>합성 곰</b>(구 · 타원체 · 캡슐로 만든 곰에 알려진 카메라로 2D 부위와
      잡음 섞인 attention 을 만든 시험, tests/test_parts_core.py)으로 정확도를 쟀다.</p>
    <div class="table-wrap"><table class="metrics">
      <tr><th rowspan="2">합성 곰 · 부위 분류 정확도<br><small>정답과 같은 부위로 분류된 복셀의 비율<br>높을수록 좋음</small></th><th colspan="2">표면 위치별</th><th colspan="2">부위별</th></tr>
      <tr><th>사진에 보이는 면</th><th>가려진 면 (옆 · 뒤)</th><th>팔</th><th>몸통</th></tr>
      <tr><td>1차 · attention + 특징 전파</td><td>85%</td><td>73%</td><td>91%</td><td>0%</td></tr>
      <tr><td>2차 · + 카메라 추정 · 보이는 복셀 투영</td><td><b>98%</b></td><td>77%</td><td>96%</td><td>25%</td></tr>
      <tr><td>3차 · + 가려진 쪽은 부피 기준</td><td><b>98%</b></td><td><b>85%</b></td><td><b>98%</b></td><td><b>48%</b></td></tr>
      <tr><td>4차 · + 특징 유사도 경계 보정</td><td><b>98%</b></td><td><b>87%</b></td><td><b>99%</b></td><td><b>52%</b></td></tr>
    </table></div>
    <p class="note">몸통은 사진에서 배에만 라벨이 있어 가장 어렵다. 4차의 향상(전체 88.57 → 90.26%, 부위 경계 복셀 76.25 → 79.95%)은 대표 특징 투표에서 나온다 —
      같은 단계를 DiT 특징으로 돌려도 거의 같았고(90.22%, 79.89%), 실제 곰의 뒷머리도 똑같이 머리로 돌아왔다.
      decoder 특징은 바뀌는 범위가 더 좁을 뿐({bfeat.get('voxels_changed', 0):,} vs 492 복셀) 더 정확하다는 근거는 아직 없다.</p>

    <h3>최종 분류 결과 — 두 에셋, 네 방향</h3>
    <p class="note">열: 정면 · 옆 · 뒤 · 반대 옆. 행: 원래 색 · attention 투표만 · 1차 · 2차의 투영 (사진에 보이는 복셀만, 회색 = 모름) · 4차 최종.
      방법은 곰에서 다듬었고, 로봇에는 완성된 4차 방법을 그대로 적용했다 (추정 카메라의 실루엣 일치 {rst['camera']['silhouette_iou']:.2f}).</p>
    <div class="row2">
      <figure class="media"><img src="assets/robot_sem_parts3d_grid.jpg" alt="안내 로봇" loading="lazy"><figcaption>안내 로봇</figcaption></figure>
      <figure class="media"><img src="assets/bear_sem_parts3d_grid.jpg" alt="곰 인형" loading="lazy"><figcaption>곰 인형</figcaption></figure>
    </div>

    <div class="callout warn">
      <b>한계와 다음 단계</b>
      <ul>
        <li><b>분류.</b> 사진에 보이는 쪽은 2D 부위를 따르지만 가려진 쪽은 부피로 추정한 것이라 곰 뒷머리처럼 조금 섞이고,
          로봇 안테나처럼 가는 부위는 일부만 잡힌다. 2D 부위 문구는 물체마다 손으로 골랐다 (곰: 'paw' 가 발에 걸려 바꿈).</li>
        <li><b>솔버.</b> 부위를 아는 XPBD(비공개 런타임에 추가)는 부피 클러스터와 형상 유지를 부위 단위로 나눈다 — 부위 기능을 끄면 기존 결과와 비트 단위로 같다.
          몸통은 덜 흔들리게 됐지만 팔만 덜렁이게 하는 것과 어깨 이음매는 아직이다. 다음은 경계 띠에서 두 형상 목표를 섞는 것(lattice shape matching),
          그리고 부위 이름에서 재질 · 질량 · 감쇠를 자동으로 정하는 것.</li>
      </ul>
    </div>
  </div>
</section>
"""


def page(rows):
    gpu = rows[0]["g3"]["gpu"].replace("NVIDIA GeForce ", "") if rows else ""
    counts = [r["g3"]["gaussians"] for r in rows] or [0]
    gmin, gmax = min(counts) // 10000, max(counts) // 10000                # 생성된 가우시안 수 (만 단위)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>권상민 · 3D Representation Portfolio</title>
<meta name="description" content="생성된 3D 를 편집·상호작용 가능한 컨텐츠로 — 3D Gaussian Splatting · Generative 3D · CUDA">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top">
  <div class="wrap nav-in">
    <a class="brand" href="#top">Sangmin Kwon</a>
    <div class="links">
      <a href="#genai">생성형 3D 데모</a><a href="#semantic">부위 → 물리</a><a href="#apg">APG-GS</a>
      <a href="#background">배경</a><a href="#next">다음 연구</a>
    </div>
  </div>
</nav>

<header class="hero" id="top">
  <div class="wrap">
    <figure class="media opener"><video src="assets/letters_drop.mp4" poster="assets/letters_drop.jpg" autoplay muted loop
      playsinline controls preload="auto"></video>
      <figcaption>TRELLIS 로 생성한 글자 3D Gaussians 에 APG-GS 물리(CUDA XPBD)를 입혀 떨어뜨렸다 (연출은 TRELLIS 티저를 참고).</figcaption></figure>
    <p class="eyebrow">Graphics · 3D Representation · Generative 3D</p>
    <h1>생성된 3D 를<br>편집하고 만질 수 있는 컨텐츠로</h1>
    <p class="lede">3D 표현(볼륨 → NeRF → 3D Gaussian Splatting)의 내부 구조를 다뤄 온 그래픽스 연구자입니다.
      렌더링만 되는 3D 를 구조를 가진 <b>편집 가능한 표현</b>으로 바꾸는 연구(APG-GS, IEEE CG&amp;A 1저자 게재 확정)를 했고,
      이번에 그 엔진을 <b>생성형 3D 파이프라인</b>에 연결했습니다.</p>
    <div class="cta">
      <a class="btn primary" href="#genai">생성형 3D 데모 보기</a>
      <a class="btn" href="{GITHUB}" target="_blank" rel="noopener">GitHub (코드)</a>
    </div>
    <div class="cards3">
      <div class="card"><span class="k">01 · 생성 → 편집</span>
        <p>텍스트 한 줄 → 이미지 → 3D Gaussians → 개수 조절(79만 → 10만, 화질 유지) → 구조 그래프 → 잡아당겨 편집.
          생성 모델 안의 신호로 가우시안마다 부위도 붙인다.</p></div>
      <div class="card"><span class="k">02 · 표현의 깊이</span>
        <p>가우시안을 점이 아닌 비등방 타원체로 보고 Bhattacharyya 겹침으로 구조를 복원, 변형 때 공분산까지 Σ′ = FΣ₀Fᵀ 로 갱신.</p></div>
      <div class="card"><span class="k">03 · 끝까지 구현</span>
        <p>CUDA XPBD 솔버(1프레임 5.7 ms · 175 FPS), OpenGL 기반 3DGS 뷰어와 편집 결과 렌더링까지 구현.</p></div>
    </div>
  </div>
</header>

<section class="fit">
  <div class="wrap">
    <h2 class="sec">Generative 2D/3D · AR 컨텐츠 저작과의 연결</h2>
    <div class="table-wrap"><table class="fit-table">
      <tr><th>필요 역량</th><th>근거</th></tr>
      <tr><td>생성형 2D/3D</td><td>SDXL-Turbo(2D) · TRELLIS(3D, structured latent → Gaussian)를 로컬 GPU 에 직접 올려 파이프라인 구성, 출력 표현을 편집 엔진에 연결 <a href="#genai">→ 데모</a></td></tr>
      <tr><td>Graphics 이론</td><td>3DGS 공분산·투영, 변형 기울기 F 추정과 Σ′ = FΣ₀Fᵀ, Bhattacharyya 거리, 볼륨 렌더링(ray marching · transfer function) <a href="#apg">→ APG-GS</a></td></tr>
      <tr><td>실시간 3D 편집 도구</td><td>SIBR(OpenGL) 뷰어에 가우시안 그래프 구축·변형·공분산 갱신 기능을 구현하고, 생성 결과를 같은 편집 파이프라인으로 연결 <a href="#apg">→ APG-GS</a></td></tr>
      <tr><td>Graphics API</td><td>OpenGL(의료 볼륨 렌더링, SIBR 뷰어), CUDA 커널 설계·최적화(워프 단위 gather, atomic 없는 결정적 누적)</td></tr>
      <tr><td>AR 컨텐츠 저작</td><td>생성한 3D 를 만질 수 있는 에셋으로: 생성 → 정리 · 정렬 → 구조 그래프 → 변형을 스크립트 하나로 잇고, 가우시안마다 부위를 붙여 부위별 물성에 사용 <a href="#semantic">→ 부위</a></td></tr>
    </table></div>
  </div>
</section>

<section id="genai">
  <div class="wrap">
    <p class="eyebrow">NEW · 생성형 3D 데모</p>
    <h2>Prompt → Editable 3D Gaussian Asset</h2>
    <p class="sub">생성 모델이 만든 3D 는 보기에는 완성돼 있지만, 가우시안끼리 관계가 없어 자연스럽게 고치거나 만질 수 없다.
      이 데모는 생성 결과에 구조를 입혀 바로 편집 가능한 에셋으로 만든다.</p>

    <ol class="pipe">
      <li><span class="tag gen">2D 생성</span><b>텍스트 → 이미지</b><small>SDXL-Turbo · 2스텝 · 512²</small></li>
      <li><span class="tag gen">3D 생성</span><b>이미지 → 3D Gaussians</b><small>TRELLIS-image-large</small></li>
      <li><span class="tag mine">직접 구현</span><b>개수 조절</b><small>중요도 선택 + 재최적화 · 79만 → 10만</small></li>
      <li><span class="tag mine">내 연구</span><b>구조 그래프</b><small>Bhattacharyya 겹침 · k-NN</small></li>
      <li><span class="tag mine">내 연구</span><b>편집 · 변형</b><small>CUDA XPBD · Σ′ = FΣ₀Fᵀ</small></li>
    </ol>

    <div class="callout">
      <b>무엇을 직접 했나</b>
      <ul>
        <li>생성 모델은 공개 사전학습 가중치를 그대로 썼다 (SDXL-Turbo, TRELLIS-image-large). 학습은 하지 않았다.</li>
        <li>Windows · {E(gpu)} 12 GB 에서 TRELLIS 를 돌리도록: 맞는 xformers 휠이 없어 희소 어텐션을 PyTorch SDPA 로 대신하는 shim 작성,
          메시 전용 의존성(kaolin, nvdiffrast) 없이 Gaussian 출력 경로만 쓰게 수정, 렌더러를 원본 3DGS 래스터라이저에 맞춤.</li>
        <li>생성된 Gaussian 을 3DGS 표준 형식·좌표계로 변환(SH 차수 · 위쪽 축 · 불투명도 정리)해 기존 APG-GS 그래프 → XPBD → 공분산 적응 파이프라인에 그대로 넣음.</li>
        <li>단계별 시간 · 메모리 · 그래프 품질 · 변형 품질을 측정 (아래 표, 전부 로그에서 읽은 값).
          코드 · 패치 · 측정 로그: <a href="{CODE}" target="_blank" rel="noopener">demos/generative-3d</a></li>
      </ul>
    </div>

    {asset_cards(rows)}

    <h3 id="reduce">개수 조절 — 10만 개로 줄여도 화질은 그대로</h3>
    <p class="sub">TRELLIS 는 표면에 걸린 복셀마다 가우시안을 <b>고정 32개</b>씩 만들어 에셋당 {gmin}만~{gmax}만 개가 된다.
      렌더링에는 괜찮지만 편집·물리에는 과하다. 중요도(불투명도 × 두 큰 축 곱) 상위 N 개만 남기면 구멍과 얼룩이 생기므로,
      원본을 무작위 시점에서 렌더한 이미지를 정답으로 남은 가우시안의 위치 · 크기 · 회전 · 불투명도 · 색을 다시 맞췄다 (Adam, L1, 3000회, 에셋당 약 13 s).</p>
    <div class="narrow">{img("distill_grid.jpg", "왼쪽 원본 · 가운데 5만 개 잘라내기만 · 오른쪽 5만 개 재최적화 (학습에 안 쓴 고정 시점)")}</div>
    <p class="note">물체 PSNR = 흰 배경을 빼고 물체 픽셀만 원본 렌더와 비교 (학습에 쓰지 않은 고정 8방향 평균). 원본은 TRELLIS 출력이므로 '원본을 얼마나 그대로 옮겼나' 를 잰 값이다.</p>
    {reduce_table(rows)}

    <h3>측정 — 생성 단계</h3>
    <p class="note">{E(gpu)} 12 GB 한 장에서 잰 값 (모두 측정 로그에서 읽음). 3D 생성 = TRELLIS 두 단계 샘플링(각 25스텝) + Gaussian 디코딩, 모델 로딩 제외.</p>
    {metrics_table(rows)}

    <div class="callout warn">
      <b>관찰과 한계</b>
      <ul>
        <li><b>생성 출력은 렌더링용으로 과하다.</b> 복셀당 고정 32개라 개수가 물체 복잡도와 무관하다. 재최적화로 줄였지만
          이것은 생성 뒤의 후처리다 — 생성 단계에서 용도(렌더 / 편집 / 모바일 AR)에 맞는 개수·크기를 직접 내놓는 것이 다음 과제.</li>
        <li><b>Σ′ 갱신의 효과는 가우시안 크기와 늘어난 정도에 달려 있다.</b> 개수를 줄여 가우시안이 커지면 위치만 갱신한 렌더와
          Σ′ 까지 갱신한 렌더의 차이가 커진다 (위 표 마지막 두 열). 화분 잎처럼 크게 늘어난 곳에서는 위치만 갱신하면 가장자리가 찢긴다.</li>
        <li><b>이미지 한 장의 한계.</b> 입력에 없던 뒷면은 생성 모델이 추정한 것이라 흐리다 (로봇 뒷면).</li>
        <li><b>그래프는 기하만 본다.</b> 겹침으로만 이어서 어느 가우시안이 팔인지 모르고, 물성도 온몸에 하나다
          → <a href="#semantic">아래</a>에서 생성 모델 안의 신호로 부위를 붙였다.</li>
      </ul>
    </div>
  </div>
</section>

{sem_section()}

<section id="apg">
  <div class="wrap">
    <p class="eyebrow">선행 연구 · IEEE Computer Graphics and Applications 1저자 (게재 확정)</p>
    <h2>APG-GS — 렌더링만 되는 3DGS 를 편집 가능한 표현으로</h2>
    <p class="sub">원본 3DGS 의 가우시안은 서로 관계가 없는 독립 primitive 라, 한 곳을 움직이면 나머지가 따라오지 않는다.
      가우시안 사이의 구조를 복원하고, 그 위에서 변형을 풀고, 렌더링 공분산까지 함께 바꾼다.</p>
    {video("prior_research_demo.mp4", "prior_research_demo.jpg", "APG-GS 실시간 변형 데모 (SIBR 뷰어)")}
    <div class="steps3">
      <div class="step"><span class="n">01</span><h4>구조 복원 — 가우시안 그래프</h4>
        <p>KD-tree 로 후보를 찾고, 두 공분산 타원체의 <b>Bhattacharyya 거리</b>로 겹침을 점수화해 강한 겹침만 연결.
          중심 거리 · 방향 · 크기를 손으로 따로 가중할 필요가 줄어든다.</p>
        <p class="eq">D<sub>B</sub> = ⅛ dᵀ Σ̄⁻¹ d + ½ ln( det Σ̄ / √(det Σᵢ det Σⱼ) ),&nbsp; Σ̄ = ½(Σᵢ + Σⱼ)</p></div>
      <div class="step"><span class="n">02</span><h4>변형 — 그래프 위 CUDA XPBD</h4>
        <p>거리 · 부피 제약을 GPU 에서 병렬로 푼다. 부피 제약을 atomic 없는 gather + 클러스터당 1워프로 바꿔
          <b>47.5 → 21.5 ms (2.21배)</b>, 워프 덕분에 가능해진 설정으로 <b>20.0 → 5.7 ms (175 FPS)</b>. 1프레임 시간, 짧을수록 좋음.</p></div>
      <div class="step"><span class="n">03</span><h4>표현 갱신 — Σ′ = FΣ₀Fᵀ</h4>
        <p>이웃 변위로 국소 변형 기울기 F 를 추정(polar 분해로 회전 · 늘어남 분리)해 가우시안의 방향과 크기도 갱신.
          중심만 옮기면 늘어난 면이 찢어진다. 비용 <b>0.4 ms · 프레임의 약 2%</b>.</p></div>
    </div>
    <div class="row3">
      {img("chair_posonly.jpg", "변형 전 (의자)")}
      {img("chair_rest.jpg", "위치만 갱신 — 늘어난 면이 찢어짐")}
      {img("chair_posshape.jpg", "위치 + Σ′ 갱신 — 면이 이어짐")}
    </div>
  </div>
</section>


<section id="background">
  <div class="wrap">
    <p class="eyebrow">연구 흐름</p>
    <h2>3D 를 어떻게 표현하고 그릴 것인가 → 어떻게 만들고 고칠 것인가</h2>
    <ol class="timeline">
      <li><b>볼륨 렌더링</b><span>의료 볼륨 데이터 · ray marching · transfer function · CUDA / OpenGL</span></li>
      <li><b>NeRF</b><span>암시적 신경 장면 표현 · 볼륨 렌더링 기반 novel view synthesis</span></li>
      <li><b>3D Gaussian Splatting</b><span>명시적 가우시안 표현 · 실시간 래스터화</span></li>
      <li><b>편집 가능한 3DGS</b><span>APG-GS (CG&amp;A) · CUDA XPBD · 공분산 갱신</span></li>
      <li class="next"><b>생성형 3D · 공간 컨텐츠</b><span>이번 데모에서 시작 → 다음 연구</span></li>
    </ol>
  </div>
</section>

<section id="next">
  <div class="wrap">
    <p class="eyebrow">다음 연구</p>
    <h2>생성된 3D 가 바로 쓰이는 공간 컨텐츠가 되도록</h2>
    <div class="cards3">
      <div class="card"><span class="k">부위 · 재질을 아는 생성</span><p>지금은 생성이 끝난 뒤 모델 안의 신호로 부위를 꺼낸다. 다음은 생성 단계에서 부위와 재질(물성)까지 함께 내놓아 바로 물리에 쓰이는 표현.</p></div>
      <div class="card"><span class="k">용도에 맞는 생성 표현</span><p>생성 가우시안의 개수 · 크기를 용도(렌더 / 물리 / 모바일 AR)에 맞게 조절하는 표현과 LOD.</p></div>
      <div class="card"><span class="k">사용자가 조종하는 저작</span><p>텍스트 · 스케치 · 드래그로 공간 컨텐츠를 만들고 고치는 도구 — 실제 공간 스캔과 생성 에셋을 한 장면에서.</p></div>
    </div>
  </div>
</section>

<footer>
  <div class="wrap">
    <p>권상민 · Sangmin Kwon — 영상은 모두 직접 녹화 · 렌더한 결과입니다. 생성 모델: SDXL-Turbo (Stability AI), TRELLIS (Microsoft), 사전학습 가중치 사용.</p>
  </div>
</footer>
</body>
</html>
"""


if __name__ == "__main__":
    rows = build_media()
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(rows))
    print(f"index.html written ({len(rows)} generated assets)")
