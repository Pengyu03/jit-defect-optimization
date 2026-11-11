"""Small protocol/mechanics checks; no production dataset fitting."""
import json
from pathlib import Path

import numpy as np
import pytest

import study_runner as runner


def toy_history():
    counts = (1, 4, 9, 15)
    history = [{'iteration': i, 'n_fun': n, 'elapsed_s': i * 0.1} for i, n in enumerate(counts)]
    snapshots = [{'iteration': i, 'x': np.array([float(i)]), 'elapsed_s': i * 0.1} for i in range(4)]
    return history, snapshots


def test_checkpoints_follow_accepted_evaluation_counts_not_iterations():
    history, snapshots = toy_history()
    selected = runner.checkpoint_grid(history, snapshots, 20, count=5)
    assert [r['target_n_fun'] for r in selected] == [0, 5, 10, 15, 20]
    assert [r['record']['iteration'] for r in selected] == [0, 1, 2, 3, 3]


def test_snapshot_compression_keeps_initial_final_and_distinct_grid_points():
    history, snapshots = toy_history()
    selected = runner.checkpoint_grid(history, snapshots, 20, count=2)
    compressed = runner.compressed_snapshots(history, snapshots, selected)
    assert [s['iteration'] for s in compressed] == [0, 3]


def test_candidate_requires_both_successful_tuning_seeds():
    good = lambda seed, score: {'seed': seed, 'validation_ap': score, 'eligible_for_tuning': True, 'status': 'evaluation_limit'}
    failed = {'seed': 2027, 'validation_ap': 0., 'eligible_for_tuning': False, 'status': 'numerical_failure'}
    candidates = {0: [good(2026, .9), failed], 1: [good(2026, .4), good(2027, .6)]}
    rows = runner.rank_candidates(candidates, candidate_count=2)
    assert rows[0]['eligible'] is False
    assert rows[0]['mean_validation_ap'] is None
    assert rows[1]['mean_validation_ap'] == pytest.approx(.5)
    assert rows[1]['rank'] == 1


def test_global_selection_uses_rank_and_deterministic_tie_break():
    screens = {}
    for dataset, ranks in zip(runner.APACHE, ([1., 2.], [2., 1.], [1., 2.])):
        screens[dataset] = [{'candidate': i, 'eligible': True, 'rank': rank} for i, rank in enumerate(ranks)]
    winner, evidence = runner.global_rank_selection(screens, candidate_count=2)
    assert winner == 0
    assert evidence[0]['mean_rank'] == pytest.approx(4/3)


def test_protocol_resume_rejects_source_or_budget_change(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'environment', lambda: {})
    protocol = {'source_sha256': {'x.py': 'first'}, 'budget': 20}
    digest = runner.ensure_protocol(tmp_path, protocol)
    assert runner.ensure_protocol(tmp_path, protocol) == digest
    with pytest.raises(ValueError, match='immutable'):
        runner.ensure_protocol(tmp_path, protocol | {'budget': 21})
    with pytest.raises(ValueError, match='immutable'):
        runner.ensure_protocol(tmp_path, protocol | {'source_sha256': {'x.py': 'changed'}})


def test_training_loader_never_reads_test_array_keys(monkeypatch):
    class RestrictedArchive:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key):
            assert key in {'X_train', 'y_train', 'X_val', 'y_val'}
            return np.array([[0.], [1.]]) if key.startswith('X') else np.array([0, 1])
    monkeypatch.setattr(runner.np, 'load', lambda *args, **kwargs: RestrictedArchive())
    loaded = runner.training_data('unused.npz')
    assert set(loaded) == {'X_train', 'y_train', 'X_val', 'y_val'}


def test_expected_manifest_has_96_unique_cases():
    cases = runner.expected_runs(Path('results/test'))
    assert len(cases) == 96
    assert len(set(cases)) == 96
    assert sum('main' in path.parts for path in cases) == 60
    assert sum('external' in path.parts for path in cases) == 24
    assert sum('width' in path.parts for path in cases) == 12


def test_prediction_metrics_do_not_silently_reuse_threshold_for_final_point():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .3, .6, .9])
    assert 'f1' in runner.prediction_metrics(y, p, .5)
    assert 'f1' not in runner.prediction_metrics(y, p, None)
    assert runner.best_f1_threshold(y, p) == pytest.approx(.6)


def test_clean_json_encodes_nonfinite_diagnostics_as_null():
    value = runner.clean_json({'failure': np.inf, 'small': np.float64(.5), 'array': np.array([np.nan])})
    assert json.loads(json.dumps(value, allow_nan=False)) == {'failure': None, 'small': .5, 'array': [None]}


def test_cli_stage_defaults_and_seed_restriction():
    main = runner.parse_args(['main'])
    assert main.datasets == runner.APACHE
    assert main.seeds == runner.MAIN_SEEDS
    assert main.max_evaluations == 1500
    external = runner.parse_args(['external'])
    assert external.datasets == runner.EXTERNAL
    assert external.seeds == runner.OTHER_SEEDS
    with pytest.raises(SystemExit):
        runner.parse_args(['main', '--seeds', '99'])
