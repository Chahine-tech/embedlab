"""The command line, and the config that refuses to be wrong quietly."""

from pathlib import Path

import pytest

from embedlab.adapters.registry import KINDS, build
from embedlab.cli.config import ConfigError, load_experiment
from embedlab.cli.main import main

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = """
name: t
dataset: {dataset}
systems:
  - name: a
    kind: bm25
"""


def test_a_minimal_experiment_loads(tmp_path):
    experiment = load_experiment(write(tmp_path, MINIMAL.format(dataset=FIXTURE)))
    assert experiment.name == "t"
    assert experiment.k == 10
    assert [system.name for system in experiment.systems] == ["a"]


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ConfigError, match="no experiment file"):
        load_experiment(tmp_path / "absent.yaml")


def test_invalid_yaml_says_so(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_experiment(write(tmp_path, "name: [unclosed\n"))


def test_a_top_level_list_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_experiment(write(tmp_path, "- a\n- b\n"))


def test_a_misspelt_option_is_refused_rather_than_ignored(tmp_path):
    """The failure this strictness exists for.

    Ignored, `stemer` hands back an unstemmed run under a stemmed name, and the
    comparison is against the wrong thing with nothing to show for it.
    """
    body = MINIMAL.format(dataset=FIXTURE) + "    stemer: snowball-en\n"
    with pytest.raises(ConfigError) as caught:
        load_experiment(write(tmp_path, body))

    message = str(caught.value)
    assert "stemer" in message
    assert "systems.0" in message, "the message must name which system is wrong"


def test_an_unknown_kind_lists_the_known_ones(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE).replace("kind: bm25", "kind: colbert")
    with pytest.raises(ConfigError) as caught:
        load_experiment(write(tmp_path, body))
    for kind in KINDS:
        assert kind in str(caught.value)


def test_a_dense_system_needs_its_model(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE).replace("kind: bm25", "kind: dense")
    with pytest.raises(ConfigError, match="model_id"):
        load_experiment(write(tmp_path, body))


def test_duplicate_names_are_refused(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE) + "  - name: a\n    kind: tfidf\n"
    with pytest.raises(ConfigError, match="duplicate system names"):
        load_experiment(write(tmp_path, body))


def test_a_baseline_must_be_one_of_the_systems(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE) + "baseline: ghost\n"
    with pytest.raises(ConfigError, match="baseline 'ghost'"):
        load_experiment(write(tmp_path, body))


def test_the_baseline_defaults_to_the_first_system(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE) + "  - name: b\n    kind: tfidf\n"
    experiment = load_experiment(write(tmp_path, body))
    assert experiment.resolved_baseline() == "a"
    assert [system.name for system in experiment.others()] == ["b"]


def test_an_explicit_baseline_wins(tmp_path):
    body = MINIMAL.format(dataset=FIXTURE) + "  - name: b\n    kind: tfidf\nbaseline: b\n"
    experiment = load_experiment(write(tmp_path, body))
    assert experiment.resolved_baseline() == "b"
    assert [system.name for system in experiment.others()] == ["a"]


def test_at_least_one_system_is_required(tmp_path):
    with pytest.raises(ConfigError, match="systems"):
        load_experiment(write(tmp_path, f"name: t\ndataset: {FIXTURE}\nsystems: []\n"))


def test_the_registry_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown retriever kind"):
        build("colbert")


def test_the_shipped_experiment_files_are_valid():
    """These are documentation; a stale one teaches the wrong shape."""
    for path in sorted((Path(__file__).parent.parent / "experiments").glob("*.yaml")):
        experiment = load_experiment(path)
        assert experiment.systems
        assert experiment.resolved_baseline() in [s.name for s in experiment.systems]


# --- the commands themselves ------------------------------------------------

pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")


def test_run_publishes_and_reports(tmp_path, capsys):
    body = MINIMAL.format(dataset=FIXTURE) + "  - name: b\n    kind: tfidf\n"
    experiment = write(tmp_path, body)
    root = tmp_path / "ws"

    assert main(["--root", str(root), "run", str(experiment)]) == 0

    output = capsys.readouterr().out
    assert "7 judged queries" in output
    assert "trust:" in output
    assert (root / "index.json").exists()
    assert len(list((root / "run").iterdir())) == 2
    assert len(list((root / "comparison").iterdir())) == 1


def test_list_then_show_describe_what_run_published(tmp_path, capsys):
    root = tmp_path / "ws"
    main(["--root", str(root), "run", str(write(tmp_path, MINIMAL.format(dataset=FIXTURE)))])
    capsys.readouterr()

    assert main(["--root", str(root), "list"]) == 0
    listed = capsys.readouterr().out
    assert "run " in listed

    run_id = next((root / "run").iterdir()).name
    assert main(["--root", str(root), "show", run_id]) == 0
    shown = capsys.readouterr().out
    assert "judged queries failed" in shown
    assert "symptom" in shown


def test_an_empty_workspace_lists_nothing(tmp_path, capsys):
    assert main(["--root", str(tmp_path / "empty"), "list"]) == 0
    assert "nothing published yet" in capsys.readouterr().out


def test_showing_an_unknown_run_fails(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "show", "ghost"]) == 1
    assert "no run 'ghost'" in capsys.readouterr().err


def test_a_broken_config_exits_non_zero(tmp_path, capsys):
    bad = write(tmp_path, "name: t\ndataset: nowhere\nsystems: []\n")
    assert main(["--root", str(tmp_path / "ws"), "run", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err
