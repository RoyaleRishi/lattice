"""The README quickstart is executable truth (M6 spec §5): extract the
first fenced python block and run it verbatim in a temp cwd."""

import re
from pathlib import Path


def test_quickstart_block_executes(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[2] / "README.md"
    assert readme.exists(), "README.md missing at repo root"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.DOTALL)
    assert blocks, "README must open with a python quickstart block"
    monkeypatch.chdir(tmp_path)  # the block writes memory.json to cwd
    namespace: dict = {}
    exec(compile(blocks[0], "README.md:quickstart", "exec"), namespace)
    assert namespace["olive"] is not None
    assert namespace["restored"].view().find_concept("olive") is not None
