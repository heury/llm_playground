"""Output paths shared by the converted GPT marimo notebooks."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo


def prepare_run(chapter: str) -> Path:
    project_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != project_dir:
        raise RuntimeError("gpt/ 디렉토리에서 uv run marimo edit <파일.py>로 실행하세요.")
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    output_dir = project_dir.parent / "outputs" / "gpt" / f"{run_id}-{chapter}-{uuid4().hex[:8]}"
    (output_dir / "datas").mkdir(parents=True)
    return output_dir
