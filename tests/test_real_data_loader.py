import pandas as pd
import pytest

from ragb.data import real_data_loader
from ragb.data.real_data_loader import RealDataUnavailableError, load_real_dataset


def _make_loader(name, succeed=True, rows=5):
    def loader():
        if not succeed:
            raise ConnectionError(f"{name} unreachable in this test")
        X = pd.DataFrame({"f0": range(rows)})
        y = pd.Series([0] * rows, name="label")
        return X, y, {"source": name}
    return loader


def test_cascade_returns_first_success(monkeypatch):
    monkeypatch.setattr(real_data_loader, "CASCADE", [
        ("a", _make_loader("a", succeed=True)),
        ("b", _make_loader("b", succeed=True)),
    ])
    X, y, meta = load_real_dataset()
    assert meta["source"] == "a"


def test_cascade_falls_through_on_failure(monkeypatch):
    monkeypatch.setattr(real_data_loader, "CASCADE", [
        ("a", _make_loader("a", succeed=False)),
        ("b", _make_loader("b", succeed=False)),
        ("c", _make_loader("c", succeed=True)),
    ])
    X, y, meta = load_real_dataset()
    assert meta["source"] == "c"


def test_preferred_skips_earlier_sources(monkeypatch):
    calls = []

    def tracking_loader(name, succeed):
        def loader():
            calls.append(name)
            if not succeed:
                raise ConnectionError("nope")
            X = pd.DataFrame({"f0": [1]})
            return X, pd.Series([0]), {"source": name}
        return loader

    monkeypatch.setattr(real_data_loader, "CASCADE", [
        ("ulb_creditcard", tracking_loader("ulb_creditcard", succeed=True)),
        ("lending_club", tracking_loader("lending_club", succeed=True)),
        ("ieee_cis", tracking_loader("ieee_cis", succeed=False)),
        ("elliptic", tracking_loader("elliptic", succeed=True)),
    ])
    X, y, meta = load_real_dataset(preferred="ieee_cis")
    assert meta["source"] == "elliptic"
    assert calls == ["ieee_cis", "elliptic"]  # earlier sources never attempted


def test_all_sources_fail_raises_clear_error_listing_each_reason(monkeypatch):
    monkeypatch.setattr(real_data_loader, "CASCADE", [
        ("a", _make_loader("a", succeed=False)),
        ("b", _make_loader("b", succeed=False)),
    ])
    with pytest.raises(RealDataUnavailableError) as exc_info:
        load_real_dataset()
    msg = str(exc_info.value)
    assert "a" in msg
    assert "b" in msg
    assert "ConnectionError" in msg


def test_unknown_preferred_raises_value_error(monkeypatch):
    monkeypatch.setattr(real_data_loader, "CASCADE", [("a", _make_loader("a"))])
    with pytest.raises(ValueError):
        load_real_dataset(preferred="not_a_real_source")
