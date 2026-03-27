from __future__ import annotations

from typer.testing import CliRunner

from biofake.cli import app
from biofake.demo import app as demo_app


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "BioFake CLI" in result.stdout


def test_demo_module_imports() -> None:
    assert hasattr(demo_app, "main")
