from backend.reid.adapter import UnavailableReIDAdapter, inspect_reid_assets


def test_reid_reports_unimplemented(tmp_settings):
    status = inspect_reid_assets(tmp_settings)
    assert status["implemented"] is False
    assert "missing_to_enable_inference" in status
    adapter = UnavailableReIDAdapter(tmp_settings)
    assert adapter.is_available() is False
    result = adapter.identify_crop(tmp_settings.project_root / "missing.jpg")
    assert result.tiger_id is None
    assert result.available is False
