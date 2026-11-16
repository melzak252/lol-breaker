"""A named CLI model must not silently resolve to a different forecast artifact."""
import sys

import pytest

from scripts import predict_matchup


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "C0-D"],
        ["--model", "C0-D-20T"],
        ["--model", "A0", "--allow-research-surrogate"],
        ["--model", "039", "--allow-research-surrogate"],
    ],
)
def test_unavailable_models_reject_before_history_acquisition(monkeypatch, arguments):
    def forbidden_history(*args, **kwargs):
        raise AssertionError("Model admission must precede history acquisition")

    monkeypatch.setattr(predict_matchup, "load_raw_matches", forbidden_history)
    monkeypatch.setattr(sys, "argv", ["predict_matchup.py", "T1", "GenG", *arguments])
    with pytest.raises(SystemExit) as failure:
        predict_matchup.main()
    assert failure.value.code == 2
