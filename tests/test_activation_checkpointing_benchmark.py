"""Safety contract for the opt-in activation-checkpoint CUDA probe."""

from __future__ import annotations

import pytest

from scripts.benchmark_activation_checkpointing import (
    OPT_IN_ENVIRONMENT_VARIABLE,
    main,
)


def test_activation_checkpoint_benchmark_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(OPT_IN_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(SystemExit, match="2"):
        main([])

    assert f"set {OPT_IN_ENVIRONMENT_VARIABLE}=1" in capsys.readouterr().err
