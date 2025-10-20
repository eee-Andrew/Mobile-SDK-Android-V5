import pytest

pytest.importorskip("numpy")

from georeg.geoid import GeoidModel


def test_geoid_fallback_zero():
    model = GeoidModel(model="FAKE-MODEL")
    assert model.ellipsoidal_to_orthometric(0.0, 0.0, 10.0) == 10.0
    assert model.orthometric_to_ellipsoidal(0.0, 0.0, 10.0) == 10.0
    assert "FAKE-MODEL" in model.model_name
