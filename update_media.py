r"""사이트 영상 하나 바꾸기: 아무 영상 → site/assets/<이름>.mp4 (H.264 · yuv420p · faststart — 브라우저 자동 재생용)
+ 포스터 <이름>.jpg. make_web.py 와 같은 인코딩 설정이라 페이지(index.html)는 고칠 필요가 없다.

  python site/update_media.py "C:\Users\OMEN PC1\Downloads\bear.mp4" gen_bear_edit --width 1280
  python site/update_media.py "C:\Users\OMEN PC1\Downloads\plant.mp4" gen_plant_edit --width 1280 --start 2 --seconds 10
  python site/update_media.py demos/generative-3d/out/letters/letters_drop.mp4 letters_drop --poster-t 4
  python site/update_media.py demos/generative-3d/out/part_shake/robot_part2_stiff.mp4 robot_part_shake --crop-top 90 --poster-t 1

ffmpeg: imageio-ffmpeg 가 있으면 그 실행 파일, 없으면 PATH 의 ffmpeg.
"""
import argparse
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("ffmpeg 이 없다: pip install imageio-ffmpeg 또는 ffmpeg 설치")
        return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="새 영상 파일")
    ap.add_argument("name", help="site/assets 안 이름 (확장자 없이) — 예: gen_bear_edit, gen_plant_edit, letters_drop")
    ap.add_argument("--width", type=int, default=0, help="가로를 이 픽셀 이하로 줄이기 (키우지는 않는다, 0 = 그대로)")
    ap.add_argument("--start", type=float, default=0.0, help="이 시각[s]부터 자르기")
    ap.add_argument("--seconds", type=float, default=0.0, help="이 길이[s]만 남기기 (0 = 끝까지)")
    ap.add_argument("--crf", type=int, default=23, help="화질 (낮을수록 좋고 파일이 커진다)")
    ap.add_argument("--crop-top", type=int, default=0, help="위쪽을 이 픽셀만큼 잘라낸다 (영상 안 제목 글자 등)")
    ap.add_argument("--poster-t", type=float, default=0.0, help="포스터로 쓸 시각 [s] (letters_drop 은 4)")
    args = ap.parse_args()
    if not os.path.isfile(args.src):
        raise SystemExit(f"없는 파일: {args.src}")
    exe = ffmpeg()
    dst = os.path.join(ASSETS, f"{args.name}.mp4")
    if not os.path.exists(dst):
        print(f"[media] 새 이름 {args.name}.mp4 — index.html 에 이 이름을 쓰는 곳이 있는지 확인할 것")
    crop = f"crop=iw:ih-{args.crop_top}:0:{args.crop_top}," if args.crop_top else ""
    vf = ["-vf", crop + (f"scale='min({args.width},iw)':-2" if args.width else "scale=trunc(iw/2)*2:trunc(ih/2)*2")]
    cut = (["-ss", f"{args.start:.3f}"] if args.start else []) + (["-t", f"{args.seconds:.3f}"] if args.seconds else [])
    tmp = dst + ".tmp.mp4"
    subprocess.run([exe, "-y", "-loglevel", "error", *cut, "-i", args.src, *vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(args.crf), "-preset", "slow", "-movflags", "+faststart", "-an", tmp], check=True)
    os.replace(tmp, dst)
    jpg = os.path.join(ASSETS, f"{args.name}.jpg")
    subprocess.run([exe, "-y", "-loglevel", "error", "-ss", f"{args.poster_t:.2f}", "-i", dst, "-frames:v", "1",
                    "-q:v", "3", jpg], check=True)
    print(f"[media] {dst} ({os.path.getsize(dst) / 2 ** 20:.1f} MB), poster {jpg}")


if __name__ == "__main__":
    main()
