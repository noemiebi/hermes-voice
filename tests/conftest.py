"""시험이 실제 데이터 폴더(~/.hermes/voice)의 에코 게이트 학습 상태를 건드리지 않게 임시 경로로 돌린다."""
import os
import tempfile

os.environ["AEC_STATE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="hv-aec-"), "aec_state.json")
