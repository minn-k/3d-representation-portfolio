"""xformers 대체 shim: TRELLIS 가 쓰는 두 함수만 torch SDPA 로 구현한다.
torch 2.5.1+cu118 에 맞는 xformers Windows 휠이 없어서 만들었다 (추론 전용)."""
__version__ = "shim-sdpa"
