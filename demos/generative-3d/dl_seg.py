"""2D 부위 분할 모델 받기: Grounding DINO (tiny, 텍스트 → 상자) + SAM (base, 상자 → 마스크)."""
from huggingface_hub import snapshot_download

for repo in ("IDEA-Research/grounding-dino-tiny", "facebook/sam-vit-base"):
    print(snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"]), flush=True)
