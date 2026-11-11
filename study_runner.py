"""Versioned multi-project study: validation-only fitting, then test evaluation.

All fitting stages access only X_train/y_train/X_val/y_val. The evaluate stage
requires the planned 96 reporting runs by default, then scores saved models.
An objective-evaluation limit is deterministic, but is not an equal-FLOP budget.
"""
from __future__ import annotations

import bootstrap

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, log_loss, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

from src.model import TanhMLP
from src.optimizers import optimize


ROOT = Path(__file__).resolve().parent
METHODS = ('gd', 'momentum', 'adam', 'lbfgs')
APACHE = ('camel', 'ignite', 'flink')
EXTERNAL = ('qt', 'openstack')
TUNING_SEEDS = (2026, 2027)
MAIN_SEEDS = (10, 11, 12, 13, 14)
OTHER_SEEDS = (10, 11, 12)
FAILURE_STATUSES = {'numerical_failure', 'line_search_failed'}
GRIDS = {
    'gd': [{'learning_rate': x} for x in (0.1, 0.3, 1., 3., 10., 30., 100., 300.)],
    'momentum': [{'learning_rate': x, 'momentum': 0.9} for x in (0.01, 0.03, 0.1, 0.3, 1., 3.)]
                + [{'learning_rate': 0.1, 'momentum': 0.95}, {'learning_rate': 0.3, 'momentum': 0.95}],
    'adam': [{'learning_rate': x} for x in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.)],
    'lbfgs': [{'history_size': x} for x in (3, 5, 10, 20, 30, 50, 75, 100)],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    files = [ROOT/'bootstrap.py', ROOT/'study_runner.py', *sorted((ROOT/'src').glob('*.py'))]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def environment():
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')},
            'threadpools': threadpool_info()}


def protocol_from_args(args):
    return {
        'schema_version': 2, 'source_sha256': source_hashes(),
        'model': {'hidden': 32, 'l2': args.l2, 'activation': 'tanh', 'penalize_all_parameters': True},
        'budget': {'main_n_fun': args.max_evaluations, 'main_time_cap_s': args.time_cap,
                   'screen_n_fun': args.screen_evaluations, 'screen_time_cap_s': args.screen_time_cap,
                   'grad_tol': 1e-4, 'checkpoint_count': 11,
                   'description': 'n_fun counts loss-only and loss+gradient calls; not equal FLOPs'},
        'grids': GRIDS, 'apache': list(APACHE), 'external': list(EXTERNAL),
        'tuning_seeds': list(TUNING_SEEDS), 'main_seeds': list(MAIN_SEEDS), 'other_seeds': list(OTHER_SEEDS),
        'widths': [8, 128], 'width_methods': ['adam', 'lbfgs'],
        'tuning_selection': 'two valid seeds; mean validation AP; top two screen candidates rerun at main budget',
        'global_selection': 'minimum mean within-project screen rank; candidate index breaks ties; no external tuning',
        'validation': '11 uniform n_fun targets; last accepted point not after target; initial used before first point',
        'ties': 'earliest checkpoint; lower candidate index',
        'test_policy': 'no test metrics in fitting; evaluate after all planned reporting runs are complete',
        'apache_status': 'pilot-informed follow-up; not a wholly untouched confirmatory test set',
    }


def ensure_protocol(out_root, protocol):
    path = Path(out_root)/'protocol.json'
    digest = fingerprint(protocol)
    if path.exists():
        existing = read_json(path)
        if existing['protocol'] != clean_json(protocol) or existing['protocol_hash'] != digest:
            raise ValueError('Protocol/source changed: choose a new --out-root; existing study is immutable')
    else:
        save_json(path, {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                         'protocol_hash': digest, 'protocol': protocol, 'environment': environment()})
    return digest


def training_data(npz_path):
    """Intentionally never access test array keys during fitting."""
    with np.load(npz_path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in ('X_train', 'y_train', 'X_val', 'y_val')}
    for split in ('train', 'val'):
        if not np.all(np.isfinite(result['X_'+split])) or set(np.unique(result['y_'+split])) != {0, 1}:
            raise ValueError('Non-finite features or missing binary class in '+split)
    return result


def checkpoint_grid(history, snapshots, max_evaluations, count=11):
    if not history:
        return []
    by_iteration = {int(s['iteration']): s for s in snapshots}
    counts = np.asarray([h['n_fun'] for h in history], dtype=np.int64)
    if np.any(np.diff(counts) < 0):
        raise ValueError('Objective counts must be nondecreasing')
    result = []
    for target in np.linspace(0., float(max_evaluations), count):
        index = max(0, int(np.searchsorted(counts, target, side='right') - 1))
        record = history[index]
        snapshot = by_iteration[int(record['iteration'])]
        result.append({'target_n_fun': float(target), 'record': record, 'snapshot': snapshot})
    return result


def best_f1_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    values = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(values))])


def select_validation(model, data, grid):
    rows, best, best_ap = [], None, -1.
    for item in grid:
        p = model.predict_proba(item['snapshot']['x'], data['X_val'])
        ap = float(average_precision_score(data['y_val'], p))
        record = item['record']
        rows.append({'target_n_fun': item['target_n_fun'], 'iteration': record['iteration'],
                     'n_fun': record['n_fun'], 'elapsed_s': record['elapsed_s'], 'validation_ap': ap})
        if ap > best_ap:
            best, best_ap = item, ap
    if best is None:
        return None, None, None, rows
    p = model.predict_proba(best['snapshot']['x'], data['X_val'])
    return best, best_ap, best_f1_threshold(data['y_val'], p), rows


def compressed_snapshots(history, snapshots, grid):
    """Keep initial/final plus grid-selected vectors, never every iterate."""
    if not history:
        return []
    wanted = {int(history[0]['iteration']), int(history[-1]['iteration'])}
    wanted.update(int(item['record']['iteration']) for item in grid)
    by_iteration = {int(s['iteration']): s for s in snapshots}
    return [by_iteration[index] for index in sorted(wanted)]


def rank_candidates(candidate_runs, candidate_count=8):
    """Two paired successful seeds are required; failures remain explicit."""
    rows = []
    for candidate in range(candidate_count):
        runs = candidate_runs[candidate]
        eligible = len(runs) == len(TUNING_SEEDS) and all(r['eligible_for_tuning'] for r in runs)
        mean_ap = float(np.mean([r['validation_ap'] for r in runs])) if eligible else None
        rows.append({'candidate': candidate, 'eligible': eligible, 'mean_validation_ap': mean_ap,
                     'statuses': [r['status'] for r in runs], 'seeds': [r['seed'] for r in runs]})
    scores = np.asarray([r['mean_validation_ap'] if r['eligible'] else -np.inf for r in rows])
    ranks = rankdata(-scores, method='average')
    for row, rank in zip(rows, ranks):
        row['rank'] = float(rank)
    return rows


def global_rank_selection(project_screens, candidate_count=8):
    """Average within-project ranks; never average raw AP across projects."""
    rows = []
    for candidate in range(candidate_count):
        components = {project: project_screens[project][candidate] for project in APACHE}
        eligible = all(value['eligible'] for value in components.values())
        rows.append({'candidate': candidate, 'eligible': eligible,
                     'mean_rank': float(np.mean([x['rank'] for x in components.values()])) if eligible else None,
                     'project_ranks': {project: value['rank'] for project, value in components.items()}})
    valid = [r for r in rows if r['eligible']]
    if not valid:
        raise RuntimeError('No candidate was valid on both tuning seeds in all Apache datasets')
    winner = min(valid, key=lambda r: (r['mean_rank'], r['candidate']))
    return winner['candidate'], rows


class Study:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out_root).resolve()
        self.data_root = Path(args.data_root).resolve()
        self.protocol = protocol_from_args(args)
        self.protocol_hash = ensure_protocol(self.out, self.protocol)
        self.cache = {}

    def dataset(self, name):
        if name not in self.cache:
            npz = self.data_root/name/'dataset.npz'
            metadata = self.data_root/name/'metadata.json'
            if not npz.exists() or not metadata.exists():
                raise FileNotFoundError(f'Missing prepared dataset: {name}')
            self.cache[name] = {'arrays': training_data(npz), 'npz_path': npz,
                                'identity': {'dataset': name, 'npz_sha256': sha256_file(npz),
                                             'metadata_sha256': sha256_file(metadata)},
                                'metadata': read_json(metadata)}
        return self.cache[name]

    def run(self, folder, dataset, method, seed, hidden, overrides, stage, max_evaluations, time_cap, selection_hash=None):
        folder = Path(folder)
        prepared = self.dataset(dataset)
        data = prepared['arrays']
        config = dict(time_budget_s=float(time_cap), max_iterations=1000000,
                      max_evaluations=int(max_evaluations), grad_tol=1e-4, snapshot_every=1, **overrides)
        identity = {'protocol_hash': self.protocol_hash, 'data': prepared['identity'],
                    'method': method, 'seed': int(seed), 'hidden': int(hidden), 'l2': self.args.l2,
                    'config': config, 'stage': stage, 'selection_hash': selection_hash}
        run_hash = fingerprint(identity)
        summary_path = folder/'summary.json'
        if summary_path.exists():
            old = read_json(summary_path)
            if old['run_hash'] != run_hash:
                raise ValueError('Refusing mixed run configuration at '+str(folder))
            if not (folder/'parameters.npz').exists() or sha256_file(folder/'parameters.npz') != old['parameters_sha256']:
                raise ValueError('Saved parameter artifact is missing or changed at '+str(folder))
            return old
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder/'run_identity.json', identity)
        model = TanhMLP(data['X_train'], data['y_train'], hidden, self.args.l2)
        x0 = model.initialize(seed)
        model.value_grad(x0)  # Shared warm-up, outside solver timing/counters.
        result = optimize(model.value_grad, model.fun, x0, method, config)
        history = result['history']
        grid = checkpoint_grid(history, result['snapshots'], max_evaluations)
        best, observed_ap, threshold, validation_rows = select_validation(model, data, grid)
        kept = compressed_snapshots(history, result['snapshots'], grid)
        valid = bool(history) and result['status'] not in FAILURE_STATUSES
        last = history[-1] if history else None
        selected_record = best['record'] if best else None
        write_csv(folder/'history.csv', history)
        write_csv(folder/'validation_checkpoints.csv', validation_rows)
        histories = {int(h['iteration']): h for h in history}
        np.savez_compressed(folder/'parameters.npz',
                            snapshots=np.stack([s['x'] for s in kept]) if kept else np.empty((0, model.size)),
                            iterations=np.asarray([s['iteration'] for s in kept], dtype=np.int64),
                            n_fun=np.asarray([histories[int(s['iteration'])]['n_fun'] for s in kept], dtype=np.int64),
                            elapsed_s=np.asarray([s['elapsed_s'] for s in kept]),
                            initial=x0, final=result['x'],
                            selected=best['snapshot']['x'] if best else x0)
        summary = dict(dataset=dataset, method=method, seed=int(seed), hidden=int(hidden),
                       l2=self.args.l2, parameters=model.size, experiment_stage=stage,
                       protocol_hash=self.protocol_hash, source_sha256=self.protocol['source_sha256'],
                       run_hash=run_hash, data_identity=prepared['identity'], config=config,
                       status=result['status'], eligible_for_tuning=valid,
                       elapsed_s=result['elapsed_s'], n_fun=result['n_fun'], n_grad=result['n_grad'],
                       diagnostics=result['diagnostics'], iterations=int(last['iteration']) if last else 0,
                       final_loss=last['loss'] if last else None,
                       final_grad_inf=last['grad_inf'] if last else None,
                       final_normalized_grad_inf=last['normalized_grad_inf'] if last else None,
                       validation_ap=observed_ap if valid else 0.0,
                       validation_ap_observed=observed_ap,
                       validation_threshold=threshold,
                       selected_iteration=selected_record['iteration'] if selected_record else None,
                       selected_elapsed_s=selected_record['elapsed_s'] if selected_record else None,
                       selected_n_fun=selected_record['n_fun'] if selected_record else None,
                       selected_loss=selected_record['loss'] if selected_record else None,
                       selected_grad_inf=selected_record['grad_inf'] if selected_record else None,
                       parameters_sha256=sha256_file(folder/'parameters.npz'),
                       completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       folder=str(folder.relative_to(self.out)),
                       test_metrics_computed=False)
        save_json(summary_path, summary)
        print(json.dumps(clean_json({key: summary[key] for key in
                         ('experiment_stage', 'dataset', 'method', 'seed', 'hidden', 'status',
                          'elapsed_s', 'n_fun', 'n_grad', 'final_loss', 'final_grad_inf', 'validation_ap')})), flush=True)
        return summary

    def tune(self, datasets):
        for dataset in datasets:
            base = self.out/'tuning'/dataset
            screens = {method: {} for method in METHODS}
            # Rotate method order by candidate; never run timed fits concurrently.
            for candidate in range(8):
                order = METHODS[candidate % 4:] + METHODS[:candidate % 4]
                for method in order:
                    screens[method][candidate] = [
                        self.run(base/method/'screen'/f'candidate{candidate}_seed{seed}', dataset,
                                 method, seed, 32, GRIDS[method][candidate], 'tune_screen',
                                 self.args.screen_evaluations, self.args.screen_time_cap)
                        for seed in TUNING_SEEDS]
            rankings = {method: rank_candidates(screens[method]) for method in METHODS}
            save_json(base/'screen_rankings.json', {'protocol_hash': self.protocol_hash, 'methods': rankings})
            selected = {}
            finalists_record = {}
            for method in METHODS:
                valid = [row for row in rankings[method] if row['eligible']]
                if not valid:
                    raise RuntimeError(f'No valid screen candidates: {dataset}/{method}')
                finalists = sorted(valid, key=lambda r: (-r['mean_validation_ap'], r['candidate']))[:2]
                final_scores = []
                for row in finalists:
                    candidate = row['candidate']
                    runs = [self.run(base/method/'finalists'/f'candidate{candidate}_seed{seed}', dataset,
                                     method, seed, 32, GRIDS[method][candidate], 'tune_finalist',
                                     self.args.max_evaluations, self.args.time_cap) for seed in TUNING_SEEDS]
                    eligible = all(run['eligible_for_tuning'] for run in runs)
                    final_scores.append({'candidate': candidate, 'eligible': eligible,
                                         'mean_validation_ap': float(np.mean([r['validation_ap'] for r in runs])) if eligible else None,
                                         'statuses': [r['status'] for r in runs]})
                eligible_finalists = [r for r in final_scores if r['eligible']]
                if not eligible_finalists:
                    raise RuntimeError(f'Both finalists failed: {dataset}/{method}; no test data was consulted')
                winner = min(eligible_finalists, key=lambda r: (-r['mean_validation_ap'], r['candidate']))
                selected[method] = {'candidate': winner['candidate'], 'config': GRIDS[method][winner['candidate']],
                                    'mean_validation_ap': winner['mean_validation_ap']}
                finalists_record[method] = final_scores
            save_json(base/'selection.json', {'protocol_hash': self.protocol_hash, 'dataset': dataset,
                       'data_identity': self.dataset(dataset)['identity'], 'methods': selected,
                       'finalists': finalists_record,
                       'selection': 'best mean validation AP over two finalist runs at main budget'})
        self.update_global_selection()

    def update_global_selection(self):
        paths = {dataset: self.out/'tuning'/dataset/'screen_rankings.json' for dataset in APACHE}
        if not all(path.exists() for path in paths.values()):
            print(json.dumps({'event': 'global_selection_pending', 'missing': [d for d,p in paths.items() if not p.exists()]}), flush=True)
            return
        screens = {d: read_json(p) for d, p in paths.items()}
        if any(screen['protocol_hash'] != self.protocol_hash for screen in screens.values()):
            raise ValueError('Global selection cannot mix protocols')
        chosen, evidence = {}, {}
        for method in METHODS:
            candidate, rows = global_rank_selection({d: screens[d]['methods'][method] for d in APACHE})
            chosen[method] = {'candidate': candidate, 'config': GRIDS[method][candidate]}
            evidence[method] = rows
        save_json(self.out/'global_selection.json', {'protocol_hash': self.protocol_hash,
                  'methods': chosen, 'rank_evidence': evidence,
                  'screen_file_sha256': {d: sha256_file(p) for d,p in paths.items()},
                  'selection': 'average Apache validation screen rank; two seeds; no external dataset tuning',
                  'limitation': 'global transfer selection uses short screen rather than finalist horizon'})

    def reporting(self, stage, datasets, seeds):
        if stage == 'external':
            selection_path = self.out/'global_selection.json'
        for dataset in datasets:
            if stage != 'external':
                selection_path = self.out/'tuning'/dataset/'selection.json'
            selection = read_json(selection_path)
            if selection['protocol_hash'] != self.protocol_hash:
                raise ValueError('Selection comes from a different protocol')
            methods = ('adam', 'lbfgs') if stage == 'width' else METHODS
            widths = (8, 128) if stage == 'width' else (32,)
            for hidden in widths:
                for index, seed in enumerate(seeds):
                    order = methods[index % len(methods):] + methods[:index % len(methods)]
                    for method in order:
                        self.run(self.out/stage/dataset/f'h{hidden}'/f'{method}_seed{seed}', dataset,
                                 method, seed, hidden, selection['methods'][method]['config'], stage,
                                 self.args.max_evaluations, self.args.time_cap, sha256_file(selection_path))
        self.write_training_manifest()

    def write_training_manifest(self):
        expected = expected_runs(self.out)
        saved = [read_json(path/'summary.json') for path in expected if (path/'summary.json').exists()]
        save_json(self.out/'training_manifest.json', {'protocol_hash': self.protocol_hash,
                  'expected_runs': len(expected), 'observed_runs': len(saved), 'complete': len(saved) == len(expected),
                  'missing_folders': [str(path.relative_to(self.out)) for path in expected if not (path/'summary.json').exists()],
                  'runs': saved})
        return saved

    def evaluate(self, allow_partial=False):
        summaries = self.write_training_manifest()
        if len(summaries) != len(expected_runs(self.out)) and not allow_partial:
            raise RuntimeError('All 96 reporting fits must finish before evaluate; no test arrays have been read')
        if not summaries:
            raise RuntimeError('No reporting fits to evaluate')
        test_cache = {}
        evaluations = []
        for summary in summaries:
            dataset = summary['dataset']
            prepared = self.dataset(dataset)
            if summary['protocol_hash'] != self.protocol_hash or summary['data_identity'] != prepared['identity']:
                raise ValueError('Data or protocol changed before evaluation')
            folder = self.out/summary['folder']
            if sha256_file(folder/'parameters.npz') != summary['parameters_sha256']:
                raise ValueError('Parameter archive changed before evaluation')
            if summary['selected_iteration'] is None:
                evaluation = {'folder': summary['folder'], 'run_hash': summary['run_hash'],
                              'training_status': summary['status'], 'test': None,
                              'reason': 'No valid accepted parameter point'}
                save_json(folder/'test_metrics.json', evaluation)
                evaluations.append(summary | {'evaluation': evaluation})
                continue
            if dataset not in test_cache:
                with np.load(prepared['npz_path'], allow_pickle=False) as data:
                    test_cache[dataset] = (np.asarray(data['X_test']), np.asarray(data['y_test']))
            X_test, y_test = test_cache[dataset]
            model = TanhMLP(prepared['arrays']['X_train'], prepared['arrays']['y_train'], summary['hidden'], self.args.l2)
            with np.load(folder/'parameters.npz', allow_pickle=False) as archive:
                p = model.predict_proba(archive['selected'], X_test)
                p_final = model.predict_proba(archive['final'], X_test)
            threshold = summary['validation_threshold']
            selected_metrics = prediction_metrics(y_test, p, threshold)
            final_metrics = prediction_metrics(y_test, p_final, None)
            np.savez_compressed(folder/'test_predictions.npz', y=y_test, p=p, p_final=p_final)
            evaluation = {'run_hash': summary['run_hash'], 'folder': summary['folder'],
                          'training_status': summary['status'], 'test': selected_metrics,
                          'test_final': final_metrics, 'n_test': len(y_test),
                          'test_positive_fraction': float(np.mean(y_test)),
                          'parameters_sha256': summary['parameters_sha256'],
                          'source_sha256': self.protocol['source_sha256']}
            save_json(folder/'test_metrics.json', evaluation)
            evaluations.append(summary | {'evaluation': evaluation})
            print(json.dumps({'event': 'test_evaluation', 'dataset': dataset, 'method': summary['method'],
                              'seed': summary['seed'], 'hidden': summary['hidden'], 'stage': summary['experiment_stage'],
                              'test_ap': selected_metrics['ap'], 'test_final_ap': final_metrics['ap']}), flush=True)
        save_json(self.out/'evaluation.json', {'protocol_hash': self.protocol_hash,
                  'expected_runs': len(expected_runs(self.out)), 'observed_runs': len(evaluations),
                  'complete': len(evaluations) == len(expected_runs(self.out)), 'runs': evaluations})


def prediction_metrics(y, probabilities, threshold=None):
    result = {'ap': float(average_precision_score(y, probabilities)),
              'roc_auc': float(roc_auc_score(y, probabilities)),
              'log_loss': float(log_loss(y, probabilities))}
    if threshold is not None:
        result.update(f1=float(f1_score(y, probabilities >= threshold, zero_division=0)), threshold=float(threshold))
    return result


def expected_runs(out):
    out = Path(out)
    result = []
    for dataset in APACHE:
        for seed in MAIN_SEEDS:
            for method in METHODS:
                result.append(out/'main'/dataset/'h32'/f'{method}_seed{seed}')
    for dataset in EXTERNAL:
        for seed in OTHER_SEEDS:
            for method in METHODS:
                result.append(out/'external'/dataset/'h32'/f'{method}_seed{seed}')
    for hidden in (8, 128):
        for seed in OTHER_SEEDS:
            for method in ('adam', 'lbfgs'):
                result.append(out/'width'/'flink'/f'h{hidden}'/f'{method}_seed{seed}')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('tune', 'main', 'external', 'width', 'evaluate'))
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/study')
    parser.add_argument('--out-root', type=Path, default=ROOT/'results/study_v2')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--max-evaluations', type=int, default=1500)
    parser.add_argument('--time-cap', type=float, default=120.)
    parser.add_argument('--screen-evaluations', type=int, default=150)
    parser.add_argument('--screen-time-cap', type=float, default=15.)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly evaluate an incomplete study (marked incomplete)')
    args = parser.parse_args(argv)
    if min(args.max_evaluations, args.screen_evaluations, args.time_cap, args.screen_time_cap, args.l2) <= 0:
        parser.error('Budgets and regularization must be positive')
    defaults = {'tune': APACHE, 'main': APACHE, 'external': EXTERNAL, 'width': ('flink',), 'evaluate': ()}
    args.datasets = tuple(args.datasets or defaults[args.stage])
    if any(dataset not in defaults[args.stage] for dataset in args.datasets):
        parser.error('Dataset does not belong to the requested stage')
    seeds = MAIN_SEEDS if args.stage == 'main' else OTHER_SEEDS
    args.seeds = tuple(args.seeds or seeds)
    if args.stage in ('main', 'external', 'width') and any(seed not in seeds for seed in args.seeds):
        parser.error('Reporting seeds must be a subset of the prespecified stage seeds')
    return args


def main(argv=None):
    args = parse_args(argv)
    with threadpool_limits(limits=1):
        study = Study(args)
        if args.stage == 'tune':
            study.tune(args.datasets)
        elif args.stage == 'evaluate':
            study.evaluate(args.allow_partial)
        else:
            study.reporting(args.stage, args.datasets, args.seeds)


if __name__ == '__main__':
    main()
