from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from scripts import prepare_torchtitan_assets as assets
from sfu_repro.torchtitan import pins


ROOT = Path(__file__).resolve().parents[1]


def test_asset_pins_and_destinations_match_torchtitan_configs() -> None:
    declared = {asset.key: asset for asset in assets.TOKENIZER_ASSETS}
    assert declared["llama3"].repository == pins.LLAMA_TOKENIZER_REPOSITORY
    assert declared["llama3"].revision == pins.LLAMA_TOKENIZER_REVISION
    assert declared["deepseek-v3"].repository == pins.DEEPSEEK_TOKENIZER_REPOSITORY
    assert declared["deepseek-v3"].revision == pins.DEEPSEEK_TOKENIZER_REVISION

    expected_by_case = {
        "b1": declared["llama3"].destination.as_posix(),
        "b2": declared["llama3"].destination.as_posix(),
        "b3": declared["llama3"].destination.as_posix(),
        "b4": declared["deepseek-v3"].destination.as_posix(),
        "b5": declared["llama3"].destination.as_posix(),
    }
    for case, expected in expected_by_case.items():
        variants = (
            ("native", "pwl2_safe_f16", "d2_safe")
            if case == "b5"
            else ("native", "polynomial")
        )
        for variant in variants:
            with (ROOT / f"configs/torchtitan/{case}_{variant}.toml").open(
                "rb"
            ) as stream:
                configured = tomllib.load(stream)["model"]["hf_assets_path"]
            assert Path(configured).as_posix().removeprefix("./") == expected


def test_default_is_offline_dry_run(monkeypatch, capsys, tmp_path: Path) -> None:
    def fail_if_loaded():
        raise AssertionError("dry run loaded huggingface_hub")

    monkeypatch.setattr(assets, "_hub_functions", fail_if_loaded)
    monkeypatch.setattr(assets, "REPOSITORY_ROOT", tmp_path)

    assert assets.main([]) == 0

    output = capsys.readouterr().out
    assert "dry run; no network access" in output
    assert pins.LLAMA_TOKENIZER_REVISION in output
    assert pins.DEEPSEEK_TOKENIZER_REVISION in output
    assert not (tmp_path / "assets").exists()


@pytest.mark.parametrize("asset", assets.TOKENIZER_ASSETS, ids=lambda item: item.key)
def test_execute_downloads_only_declared_tokenizer_files_and_hashes_them(
    asset: assets.TokenizerAsset,
    tmp_path: Path,
) -> None:
    listed_calls: list[dict[str, str]] = []
    download_calls: list[dict[str, str]] = []

    def fake_list_repo_files(**kwargs):
        listed_calls.append(kwargs)
        return [
            *asset.files,
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
        ]

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"contents:{kwargs['filename']}".encode())
        return str(destination)

    manifest_path = assets.prepare_asset(
        asset,
        repository_root=tmp_path,
        list_repo_files=fake_list_repo_files,
        hf_hub_download=fake_download,
    )

    expected_source = {"repo_id": asset.repository, "revision": asset.revision}
    assert listed_calls == [expected_source]
    assert [call["filename"] for call in download_calls] == list(asset.files)
    assert all(call["repo_id"] == asset.repository for call in download_calls)
    assert all(call["revision"] == asset.revision for call in download_calls)
    assert all("token" not in call for call in download_calls)
    assert not any(
        call["filename"].endswith((".safetensors", ".bin", ".pth"))
        for call in download_calls
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["asset_type"] == "tokenizer"
    assert manifest["repository"] == asset.repository
    assert manifest["revision"] == asset.revision
    assert [record["path"] for record in manifest["files"]] == list(asset.files)
    for record in manifest["files"]:
        content = f"contents:{record['path']}".encode()
        assert record["sha256"] == hashlib.sha256(content).hexdigest()


def test_missing_declared_file_fails_before_any_download(tmp_path: Path) -> None:
    asset = assets.TOKENIZER_ASSETS[0]
    downloaded = False

    def fake_download(**kwargs):
        nonlocal downloaded
        downloaded = True

    with pytest.raises(assets.AssetPreparationError, match="missing declared"):
        assets.prepare_asset(
            asset,
            repository_root=tmp_path,
            list_repo_files=lambda **kwargs: asset.files[:-1],
            hf_hub_download=fake_download,
        )

    assert not downloaded
    assert not (tmp_path / asset.destination).exists()


def test_failed_update_invalidates_an_existing_manifest(tmp_path: Path) -> None:
    asset = assets.TOKENIZER_ASSETS[0]
    destination = tmp_path / asset.destination
    destination.mkdir(parents=True)
    manifest = destination / assets.MANIFEST_NAME
    manifest.write_text("stale", encoding="utf-8")

    def fail_download(**kwargs):
        raise RuntimeError("provider failure")

    with pytest.raises(assets.AssetPreparationError, match="Hub download failed"):
        assets.prepare_asset(
            asset,
            repository_root=tmp_path,
            list_repo_files=lambda **kwargs: asset.files,
            hf_hub_download=fail_download,
        )

    assert not manifest.exists()


def test_credentials_are_not_accepted_or_echoed(capsys) -> None:
    sentinel_value = "hf_do-not-print-this-value"
    with pytest.raises(SystemExit) as error:
        assets.parse_args(["--hf-token", sentinel_value])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert sentinel_value not in captured.out
    assert sentinel_value not in captured.err
