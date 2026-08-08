"""Safety contract for the opt-in FlashAttention CUDA probe."""

from __future__ import annotations

import pytest

from scripts.benchmark_flash_attention import (
    OPT_IN_ENVIRONMENT_VARIABLE,
    main,
)


def test_flash_benchmark_requires_explicit_local_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(OPT_IN_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(SystemExit, match="2"):
        main([])

    assert f"set {OPT_IN_ENVIRONMENT_VARIABLE}=1" in capsys.readouterr().err
