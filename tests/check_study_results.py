"""Replay the completed 96-run study without fitting any models.

Run only after timed training and study_runner.py evaluate have completed:
  python tests/check_study_results.py --run results/study_v2 \
      --data-root data/study --out tests/study_integrity.json

The checker re-evaluates stored parameter vectors. Most intermediate training
vectors were intentionally not saved, so their objective/gradient values cannot
be independently replayed; their CSV structure and counters are checked only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, log_loss, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits


METHODS = ('gd', 'momentum', 'adam', 'lbfgs')
APACHE = ('camel', 'ignite', 'flink')
EXTERNAL = ('qt', 'openstack')
FAILURES = {'numerical_failure', 'line_search_failed'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def expected_cases():
    cases = []
    for dataset in APACHE:
        for seed in range(10, 15):
            cases.extend(('main', dataset, 32, method, seed) for method in METHODS)
    for dataset in EXTERNAL:
        for seed in range(10, 13):
            cases.extend(('external', dataset, 32, method, seed) for method in METHODS)
    for hidden in (8, 128):
        for seed in range(10, 13):
            cases.extend(('width', 'flink', hidden, method, seed) for method in ('adam', 'lbfgs'))
    return cases


def case_key(summary):
    return (summary['experiment_stage'], summary['dataset'], int(summary['hidden']), summary['method'], int(summary['seed']))


def case_folder(case):
    stage, dataset, hidden, method, seed = case
    return Path(stage)/dataset/f'h{hidden}'/f'{method}_seed{seed}'


class Recorder:
    def __init__(self):
        self.errors = []
        self.check_count = 0
        self.max_abs_error = {}
        self.audited_files_sha256 = {}

    def audit_file(self, path, required=True):
        """Bind this check to the exact input bytes using portable project paths."""
        path = Path(path).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            self.check(False, f'Audit input is outside the project and cannot be packaged: {path}')
            return None
        if not path.is_file():
            if required:
                self.check(False, f'Required audited input is absent: {relative}')
            return None
        digest = file_hash(path)
        previous = self.audited_files_sha256.get(relative)
        if previous is not None:
            self.check(previous == digest, f'Audited input unchanged between reads: {relative}')
        else:
            self.audited_files_sha256[relative] = digest
        return digest

    def verify_audited_files_unchanged(self):
        """Reject an input changed while replay was in progress."""
        for relative, digest in self.audited_files_sha256.items():
            path = ROOT/relative
            self.check(path.is_file() and file_hash(path) == digest,
                       f'Audited input unchanged at audit completion: {relative}')

    def check(self, condition, label):
        self.check_count += 1
        if not bool(condition):
            self.errors.append(label)
        return bool(condition)

    def close(self, actual, saved, category, label, atol=1e-10, rtol=1e-9):
        self.check_count += 1
        if saved is None:
            self.errors.append(label+': saved value is null')
            return False
        actual = np.asarray(actual, dtype=np.float64)
        saved = np.asarray(saved, dtype=np.float64)
        if actual.shape != saved.shape:
            self.errors.append(label+': shape mismatch')
            return False
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(saved)):
            self.errors.append(label+': non-finite value')
            return False
        error = float(np.max(np.abs(actual-saved))) if actual.size else 0.
        self.max_abs_error[category] = max(error, self.max_abs_error.get(category, 0.))
        good = bool(np.allclose(actual, saved, atol=atol, rtol=rtol))
        if not good:
            self.errors.append(f'{label}: max_abs_error={error:.17g}')
        return good


def initial_parameters(d, h, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.uniform(-np.sqrt(6./(d+h)), np.sqrt(6./(d+h)), d*h),
                           np.zeros(h), rng.uniform(-np.sqrt(6./(h+1)), np.sqrt(6./(h+1)), h), [0.]))


def unpack(theta, d, h):
    k = d*h
    return theta[:k].reshape(d, h), theta[k:k+h], theta[k+h:k+2*h], theta[-1]


def probabilities(theta, X, h):
    W, b, v, c = unpack(theta, X.shape[1], h)
    return expit(np.tanh(X@W+b)@v+c)


def replay_objective(theta, X, y, h, l2):
    """Independent formula implementation; does not call the training model."""
    W, b, v, c = unpack(theta, X.shape[1], h)
    H = np.tanh(X@W+b)
    z = H@v+c
    positive_count = int(np.sum(y))
    negative_count = len(y)-positive_count
    weights = np.where(y == 1, negative_count/positive_count, 1.)
    weights /= weights.sum()
    loss = float(np.sum(weights*(np.logaddexp(0., z)-y*z)) + .5*l2*np.dot(theta, theta))
    errors = weights*(expit(z)-y)
    hidden_errors = errors[:, None]*(1.-H**2)*v[None, :]
    gradient = np.empty_like(theta, dtype=np.float64)
    k = X.shape[1]*h
    gradient[:k] = (X.T@hidden_errors).reshape(-1)
    gradient[k:k+h] = np.sum(hidden_errors, axis=0)
    gradient[k+h:k+2*h] = np.sum(H*errors[:, None], axis=0)
    gradient[-1] = np.sum(errors)
    gradient += l2*theta
    return loss, gradient


def f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    scores = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(scores))])


def replay_metrics(y, p, threshold=None):
    result = {'ap': float(average_precision_score(y, p)),
              'roc_auc': float(roc_auc_score(y, p)), 'log_loss': float(log_loss(y, p))}
    if threshold is not None:
        result.update(f1=float(f1_score(y, p>=threshold, zero_division=0)), threshold=float(threshold))
    return result


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def numeric_history(path):
    rows = csv_rows(path)
    integers = {'iteration', 'n_fun', 'n_grad'}
    return [{key: int(float(value)) if key in integers else float(value) for key, value in row.items()} for row in rows]


def resolve_raw(metadata, data_name):
    recorded = Path(metadata['raw_path'])
    if recorded.exists():
        return recorded
    fallback = ROOT/'data'/('external_raw' if data_name in EXTERNAL else 'raw')/recorded.name
    return fallback


def load_dataset(name, data_root, recorder):
    folder = data_root/name
    recorder.audit_file(folder/'metadata.json')
    recorder.audit_file(folder/'dataset.npz')
    metadata = read_json(folder/'metadata.json')
    raw_path = resolve_raw(metadata, name)
    recorder.audit_file(raw_path)
    recorder.check(raw_path.exists(), f'{name}: raw data exists')
    raw_sha = file_hash(raw_path) if raw_path.exists() else None
    recorder.check(raw_sha == metadata['raw_sha256'], f'{name}: raw SHA-256 matches metadata')
    identity = {'dataset': name, 'npz_sha256': file_hash(folder/'dataset.npz'),
                'metadata_sha256': file_hash(folder/'metadata.json')}
    with np.load(folder/'dataset.npz', allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    for split in ('train', 'val', 'test'):
        X, y = data['X_'+split], data['y_'+split]
        recorder.check(np.all(np.isfinite(X)) and np.all(np.isfinite(y)), f'{name}/{split}: finite data')
        recorder.check(set(np.unique(y)) == {0, 1}, f'{name}/{split}: both binary classes')
        recorder.check(len(X) == len(y) == metadata['splits'][split]['n'], f'{name}/{split}: count matches metadata')
        recorder.check(int(y.sum()) == metadata['splits'][split]['positive'], f'{name}/{split}: class count matches metadata')
    ids = {s: set(data[s+'_ids'].tolist()) for s in ('train', 'val', 'test')}
    recorder.check(not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']), f'{name}: disjoint IDs')
    recorder.check(data['timestamps_train'].max() < data['timestamps_val'].min() < data['timestamps_test'].min(), f'{name}: chronological split starts')
    recorder.check(data['timestamps_val'].max() < data['timestamps_test'].min(), f'{name}: validation precedes test')
    return {'arrays': data, 'identity': identity, 'metadata': metadata,
            'raw_path_checked': str(raw_path), 'raw_sha256': raw_sha}


def replay_run(case, summary, evaluation, run_root, prepared, protocol, protocol_hash, recorder, initial_cache):
    name = '/'.join(map(str, case))
    before_errors, before_checks = len(recorder.errors), recorder.check_count
    folder = run_root/case_folder(case)
    for filename in ('summary.json', 'run_identity.json', 'parameters.npz', 'test_metrics.json'):
        recorder.audit_file(folder/filename)
    # No-history failures legitimately omit these files; their absence is
    # checked below, and all files that do exist are bound to this audit.
    for filename in ('history.csv', 'validation_checkpoints.csv', 'test_predictions.npz'):
        recorder.audit_file(folder/filename, required=False)
    local_summary = read_json(folder/'summary.json')
    recorder.check(summary == local_summary, name+': manifest and summary identical')
    recorder.check(case_key(summary) == case, name+': expected key')
    recorder.check(Path(summary['folder']) == case_folder(case), name+': expected folder')
    recorder.check(summary['protocol_hash'] == protocol_hash, name+': protocol hash')
    recorder.check(summary['source_sha256'] == protocol['source_sha256'], name+': source hashes')
    recorder.check(summary['data_identity'] == prepared['identity'], name+': NPZ/metadata hashes')
    identity = read_json(folder/'run_identity.json')
    recorder.check(object_hash(identity) == summary['run_hash'], name+': run identity digest')
    recorder.check(identity['protocol_hash'] == protocol_hash, name+': identity protocol')
    recorder.check(identity['data'] == prepared['identity'], name+': identity data')
    recorder.check(identity['config'] == summary['config'], name+': identity config')
    selection_path = run_root/'global_selection.json' if case[0] == 'external' else run_root/'tuning'/case[1]/'selection.json'
    recorder.audit_file(selection_path)
    selection = read_json(selection_path)
    recorder.check(file_hash(selection_path) == identity['selection_hash'], name+': selection file hash')
    recorder.check(selection['protocol_hash'] == protocol_hash, name+': selection protocol')
    for setting, value in selection['methods'][case[3]]['config'].items():
        recorder.check(summary['config'][setting] == value, name+': frozen setting '+setting)
    recorder.check(summary['config']['max_evaluations'] == protocol['budget']['main_n_fun'], name+': fixed evaluation limit')
    recorder.check(summary['config']['time_budget_s'] == protocol['budget']['main_time_cap_s'], name+': fixed safety cap')
    recorder.check(summary['config']['grad_tol'] == 1e-4, name+': unchanged stationarity criterion')
    recorder.check(summary['test_metrics_computed'] is False, name+': fit summary did not include test metrics')
    recorder.check('test' not in summary and 'test_final' not in summary, name+': test separated from fit summary')
    parameter_sha = file_hash(folder/'parameters.npz')
    recorder.check(parameter_sha == summary['parameters_sha256'], name+': parameter archive hash')
    with np.load(folder/'parameters.npz', allow_pickle=False) as archive:
        parameters = {key: np.asarray(archive[key]) for key in archive.files}
    data = prepared['arrays']
    X, y, h, l2 = data['X_train'], data['y_train'], case[2], float(summary['l2'])
    expected_initial = initial_parameters(X.shape[1], h, case[4])
    recorder.check(np.array_equal(parameters['initial'], expected_initial), name+': seed initialization reproduced exactly')
    paired_key = (case[1], h, case[4])
    if paired_key in initial_cache:
        recorder.check(np.array_equal(initial_cache[paired_key], parameters['initial']), name+': paired-method initial parameters exactly equal')
    else:
        initial_cache[paired_key] = parameters['initial'].copy()
    recorder.check(parameters['initial'].size == summary['parameters'] == X.shape[1]*h+2*h+1, name+': parameter dimension')
    history = numeric_history(folder/'history.csv')
    local_evaluation = read_json(folder/'test_metrics.json')
    recorder.check(evaluation == local_evaluation, name+': evaluation manifest and local file identical')
    recorder.check(evaluation['run_hash'] == summary['run_hash'], name+': evaluation run identity')
    recorder.check(evaluation['training_status'] == summary['status'], name+': failure/stop status retained')

    if not history:
        recorder.check(summary['selected_iteration'] is None, name+': no-history selection explicitly absent')
        recorder.check(summary['eligible_for_tuning'] is False, name+': no-history run is ineligible')
        recorder.check(summary['validation_ap'] == 0., name+': failed tuning score is worst score')
        recorder.check(summary['final_loss'] is None and summary['final_grad_inf'] is None, name+': absent final diagnostics are null')
        recorder.check(parameters['snapshots'].shape[0] == 0, name+': no accepted snapshots invented')
        recorder.check(np.array_equal(parameters['selected'], parameters['initial']), name+': explicitly recorded initial fallback')
        recorder.check(np.array_equal(parameters['final'], parameters['initial']), name+': no accepted update in final fallback')
        recorder.check(summary['validation_ap_observed'] is None and summary['validation_threshold'] is None, name+': no validation result invented without accepted points')
        recorder.check(not csv_rows(folder/'validation_checkpoints.csv'), name+': no checkpoint rows invented without accepted points')
        recorder.check(evaluation['test'] is None, name+': no test result invented for no-history run')
        return {'case': list(case), 'training_status': summary['status'], 'has_accepted_points': False,
                'checks': recorder.check_count-before_checks, 'errors': recorder.errors[before_errors:],
                'passed': len(recorder.errors) == before_errors,
                'scope': 'Failure/fallback representation checked; no valid objective or predictive result was claimed.'}

    recorder.check([row['iteration'] for row in history] == list(range(len(history))), name+': accepted iteration sequence')
    counts = np.asarray([row['n_fun'] for row in history])
    gradients = np.asarray([row['n_grad'] for row in history])
    times = np.asarray([row['elapsed_s'] for row in history])
    recorder.check(np.all(np.diff(counts)>=0) and np.all(np.diff(gradients)>=0), name+': monotonic counters')
    recorder.check(np.all(np.diff(times)>=0), name+': monotonic accepted-point time')
    recorder.check(np.all(gradients <= counts), name+': gradient computations included in objective counts')
    recorder.check(history[-1]['n_fun'] <= summary['n_fun'] <= summary['config']['max_evaluations'], name+': accepted/total/maximum objective counts')
    recorder.check(history[-1]['n_grad'] <= summary['n_grad'] <= summary['n_fun'], name+': total gradient count consistency')
    recorder.check(history[-1]['elapsed_s'] <= summary['elapsed_s'], name+': accepted point precedes exit time')
    recorder.check(history[-1]['iteration'] == summary['iterations'], name+': final iteration')
    finite_fields = [row[field] for row in history for field in ('loss','grad_norm','grad_inf','normalized_grad_inf','elapsed_s')]
    recorder.check(np.all(np.isfinite(finite_fields)), name+': no non-finite accepted diagnostics')
    snapshot_indices = parameters['iterations'].astype(int).tolist()
    recorder.check(snapshot_indices == sorted(set(snapshot_indices)), name+': unique ordered saved iterations')
    snapshots = {index: theta for index, theta in zip(snapshot_indices, parameters['snapshots'])}
    recorder.check(0 in snapshots and history[-1]['iteration'] in snapshots, name+': initial and final snapshots present')
    recorder.check(np.array_equal(snapshots[0], parameters['initial']), name+': initial snapshot matches initial vector')
    recorder.check(np.array_equal(snapshots[history[-1]['iteration']], parameters['final']), name+': final snapshot matches final vector')
    by_iteration = {row['iteration']: row for row in history}
    for index, n_fun, elapsed in zip(snapshot_indices, parameters['n_fun'], parameters['elapsed_s']):
        recorder.check(int(n_fun) == by_iteration[index]['n_fun'], name+': saved snapshot count '+str(index))
        recorder.close(float(elapsed), by_iteration[index]['elapsed_s'], 'saved_time', name+': saved snapshot time '+str(index))

    initial_loss, initial_gradient = replay_objective(parameters['initial'], X, y, h, l2)
    gradient_scale = max(1., float(np.max(np.abs(initial_gradient))))
    recorder.close(initial_loss, history[0]['loss'], 'objective', name+': initial loss')
    recorder.close(float(np.max(np.abs(initial_gradient))), history[0]['grad_inf'], 'gradient_inf', name+': initial gradient')
    final_loss, final_gradient = replay_objective(parameters['final'], X, y, h, l2)
    final_inf = float(np.max(np.abs(final_gradient)))
    recorder.close(final_loss, summary['final_loss'], 'objective', name+': final summary loss')
    recorder.close(final_loss, history[-1]['loss'], 'objective', name+': final history loss')
    recorder.close(final_inf, summary['final_grad_inf'], 'gradient_inf', name+': final summary gradient')
    recorder.close(final_inf, history[-1]['grad_inf'], 'gradient_inf', name+': final history gradient')
    recorder.close(float(np.linalg.norm(final_gradient)), history[-1]['grad_norm'], 'gradient_norm', name+': final gradient norm')
    recorder.close(final_inf/gradient_scale, summary['final_normalized_grad_inf'], 'stationarity_residual', name+': final summary residual')
    recorder.close(final_inf/gradient_scale, history[-1]['normalized_grad_inf'], 'stationarity_residual', name+': final history residual')
    if summary['status'] == 'converged':
        recorder.check(final_inf/gradient_scale <= 1e-4*(1+1e-8), name+': convergence status meets unchanged threshold')
    valid_for_tuning = summary['status'] not in FAILURES
    recorder.check(summary['eligible_for_tuning'] == valid_for_tuning, name+': failure status determines eligibility')

    validation_rows = csv_rows(folder/'validation_checkpoints.csv')
    recorder.check(len(validation_rows) == 11, name+': exactly 11 validation opportunities')
    targets = np.linspace(0., float(summary['config']['max_evaluations']), 11)
    chosen_indices, observed_scores, probability_cache = [], [], {}
    for position, target in enumerate(targets):
        eligible_indices = [i for i, row in enumerate(history) if row['n_fun'] <= target]
        position_in_history = eligible_indices[-1] if eligible_indices else 0
        point = history[position_in_history]
        iteration = point['iteration']
        chosen_indices.append(iteration)
        recorder.check(iteration in snapshots, name+': grid-selected snapshot retained '+str(iteration))
        row = validation_rows[position]
        recorder.close(float(target), float(row['target_n_fun']), 'checkpoint_target', name+': checkpoint target '+str(position))
        recorder.check(int(row['iteration']) == iteration and int(row['n_fun']) == point['n_fun'], name+': last-accepted grid point '+str(position))
        recorder.close(float(row['elapsed_s']), point['elapsed_s'], 'saved_time', name+': validation snapshot time')
        if iteration not in probability_cache:
            probability_cache[iteration] = probabilities(snapshots[iteration], data['X_val'], h)
        ap = float(average_precision_score(data['y_val'], probability_cache[iteration]))
        observed_scores.append(ap)
        recorder.close(ap, float(row['validation_ap']), 'validation_ap', name+': validation AP '+str(position))
    expected_kept = set(chosen_indices) | {0, history[-1]['iteration']}
    recorder.check(set(snapshot_indices) == expected_kept, name+': compression retains exactly required points')
    best_position = int(np.argmax(observed_scores))  # first maximum preserves earliest tie.
    best_iteration = chosen_indices[best_position]
    recorder.check(summary['selected_iteration'] == best_iteration, name+': selected iteration and earliest tie')
    recorder.check(np.array_equal(parameters['selected'], snapshots[best_iteration]), name+': selected vector matches selected snapshot')
    recorder.close(observed_scores[best_position], summary['validation_ap_observed'], 'validation_ap', name+': observed selected validation AP')
    recorder.close(observed_scores[best_position] if valid_for_tuning else 0., summary['validation_ap'], 'validation_ap_score', name+': failure-aware selection score')
    threshold = f1_threshold(data['y_val'], probability_cache[best_iteration])
    recorder.close(threshold, summary['validation_threshold'], 'validation_threshold', name+': F1 threshold recomputed only from validation')
    selected_loss, selected_gradient = replay_objective(parameters['selected'], X, y, h, l2)
    recorder.close(selected_loss, summary['selected_loss'], 'objective', name+': selected summary loss')
    recorder.close(selected_loss, by_iteration[best_iteration]['loss'], 'objective', name+': selected history loss')
    selected_inf = float(np.max(np.abs(selected_gradient)))
    recorder.close(selected_inf, summary['selected_grad_inf'], 'gradient_inf', name+': selected summary gradient')
    recorder.close(selected_inf, by_iteration[best_iteration]['grad_inf'], 'gradient_inf', name+': selected history gradient')
    recorder.close(selected_inf/gradient_scale, by_iteration[best_iteration]['normalized_grad_inf'], 'stationarity_residual', name+': selected history residual')
    recorder.check(summary['selected_n_fun'] == by_iteration[best_iteration]['n_fun'], name+': selected count')
    recorder.close(summary['selected_elapsed_s'], by_iteration[best_iteration]['elapsed_s'], 'saved_time', name+': selected time')

    y_test = data['y_test']
    p = probabilities(parameters['selected'], data['X_test'], h)
    p_final = probabilities(parameters['final'], data['X_test'], h)
    with np.load(folder/'test_predictions.npz', allow_pickle=False) as saved:
        recorder.check(np.array_equal(saved['y'], y_test), name+': test labels preserved exactly')
        recorder.close(p, saved['p'], 'test_probability', name+': selected test probabilities')
        recorder.close(p_final, saved['p_final'], 'test_final_probability', name+': final test probabilities')
    for field, value in replay_metrics(y_test, p, threshold).items():
        recorder.close(value, evaluation['test'][field], 'test_'+field, name+': selected test '+field)
    for field, value in replay_metrics(y_test, p_final).items():
        recorder.close(value, evaluation['test_final'][field], 'test_final_'+field, name+': final test '+field)
    recorder.check('f1' not in evaluation['test_final'], name+': final point does not reuse selected F1 threshold')
    recorder.check(evaluation['n_test'] == len(y_test), name+': evaluation test count')
    recorder.close(float(np.mean(y_test)), evaluation['test_positive_fraction'], 'test_prevalence', name+': test class fraction')
    recorder.check(evaluation['parameters_sha256'] == parameter_sha, name+': evaluated parameter checksum')
    recorder.check(evaluation['source_sha256'] == protocol['source_sha256'], name+': evaluation source checksum')
    return {'case': list(case), 'training_status': summary['status'], 'has_accepted_points': True,
            'accepted_history_rows': len(history), 'saved_parameter_points': len(snapshot_indices),
            'replayed_validation_opportunities': len(validation_rows),
            'selected_iteration': best_iteration, 'final_iteration': history[-1]['iteration'],
            'final_loss_recomputed': final_loss, 'final_residual_recomputed': final_inf/gradient_scale,
            'checks': recorder.check_count-before_checks, 'errors': recorder.errors[before_errors:],
            'passed': len(recorder.errors) == before_errors,
            'scope': 'Initial/final/selected objectives and gradients; all 11 validation grid predictions; selected/final test predictions and metrics. Other unsaved training vectors cannot be replayed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/study_v2')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/study')
    parser.add_argument('--out', type=Path, default=ROOT/'tests/study_integrity.json')
    args = parser.parse_args()
    run_root, data_root = args.run.resolve(), args.data_root.resolve()
    recorder = Recorder()
    recorder.audit_file(Path(__file__))
    for filename in ('protocol.json', 'training_manifest.json', 'evaluation.json'):
        recorder.audit_file(run_root/filename)
    protocol_record = read_json(run_root/'protocol.json')
    protocol, protocol_hash = protocol_record['protocol'], protocol_record['protocol_hash']
    recorder.check(object_hash(protocol) == protocol_hash, 'Protocol content hash matches its recorded digest')
    current_source_files = [ROOT/'bootstrap.py', ROOT/'study_runner.py', *sorted((ROOT/'src').glob('*.py'))]
    for path in current_source_files:
        recorder.audit_file(path)
    current_source = {str(path.relative_to(ROOT)): file_hash(path) for path in current_source_files}
    recorder.check(current_source == protocol['source_sha256'], 'Current training-source namespace and SHA-256 exactly match frozen protocol')
    training = read_json(run_root/'training_manifest.json')
    scored = read_json(run_root/'evaluation.json')
    recorder.check(training['protocol_hash'] == scored['protocol_hash'] == protocol_hash, 'All manifests share the frozen protocol')
    expected = set(expected_cases())
    training_keys = [case_key(row) for row in training['runs']]
    evaluation_keys = [case_key(row) for row in scored['runs']]
    recorder.check(len(expected) == 96, 'Checker plan has 96 unique cases')
    recorder.check(len(training_keys) == len(set(training_keys)) == 96 and set(training_keys) == expected, 'Training manifest is complete and unique')
    recorder.check(len(evaluation_keys) == len(set(evaluation_keys)) == 96 and set(evaluation_keys) == expected, 'Evaluation manifest is complete and unique')
    recorder.check(training['complete'] is True and scored['complete'] is True, 'Both manifests claim complete reporting runs')
    recorder.check(training['expected_runs'] == training['observed_runs'] == scored['expected_runs'] == scored['observed_runs'] == 96, 'Manifest run counts agree')
    training_map = {case_key(row): row for row in training['runs']}
    evaluation_map = {case_key(row): row for row in scored['runs']}
    datasets = {}
    runs, initial_cache = [], {}
    with threadpool_limits(limits=1):
        for dataset in APACHE+EXTERNAL:
            try:
                datasets[dataset] = load_dataset(dataset, data_root, recorder)
            except Exception as exc:
                recorder.errors.append(f'Dataset {dataset} cannot be checked: {type(exc).__name__}: {exc}')
        for case in expected_cases():
            name = '/'.join(map(str, case))
            try:
                summary = training_map[case]
                scored_row = evaluation_map[case]
                recorder.check({k:v for k,v in scored_row.items() if k != 'evaluation'} == summary, name+': evaluation retains original fitting summary')
                result = replay_run(case, summary, scored_row['evaluation'], run_root, datasets[case[1]],
                                    protocol, protocol_hash, recorder, initial_cache)
            except Exception as exc:
                message = f'{name}: replay exception {type(exc).__name__}: {exc}'
                recorder.errors.append(message)
                result = {'case': list(case), 'passed': False, 'errors': [message],
                          'training_status': training_map.get(case, {}).get('status', 'missing')}
            runs.append(result)
            print(json.dumps({'case': list(case), 'passed': result['passed'],
                              'training_status': result['training_status']}), flush=True)
    recorder.verify_audited_files_unchanged()
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': not recorder.errors, 'planned_runs': 96, 'checked_runs': len(runs),
        'protocol_hash': protocol_hash, 'run_directory': str(run_root),
        'check_count': recorder.check_count, 'errors': recorder.errors,
        'audited_files_sha256': dict(sorted(recorder.audited_files_sha256.items())),
        'audited_paths_relative_to': 'project root; forward-slash separators',
        'tolerances': {'absolute': 1e-10, 'relative': 1e-9, 'initial_vectors': 'exact elementwise equality'},
        'maximum_absolute_errors': recorder.max_abs_error,
        'training_status_counts': dict(Counter(row['training_status'] for row in runs)),
        'data_checks': {name: {'identity': value['identity'], 'raw_sha256': value['raw_sha256'],
                               'raw_path_checked': value['raw_path_checked']} for name,value in datasets.items()},
        'initial_pair_groups': len(initial_cache),
        'scope': [
            'No model fitting, optimizer steps or test-based selection were performed by this checker.',
            'Frozen source namespace, protocol, raw-data, prepared-data and parameter hashes were checked.',
            'audited_files_sha256 binds this checker and every accessed study input to the exact bytes checked; inputs are rehashed at audit completion. The audit report itself is excluded to avoid a circular checksum.',
            'Independent objective/gradient formulas replay initial, selected and final parameter vectors.',
            'Every one of the 11 stored evaluation-count validation opportunities is replayed, including repeated points.',
            'All reported selected/final test probabilities and metrics are recomputed from saved vectors.',
            'Intermediate training vectors outside the compressed snapshot set were not saved; their objective/gradient values cannot be independently replayed. Their history ordering, counts and finite diagnostics are checked.',
            'Failures are retained explicitly; an initial fallback without accepted points must have no invented test result.',
            'Apache follow-up remains pilot-informed. Integrity checks do not make previously inspected pilot data a blind test.'
        ],
        'runs': runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': recorder.check_count,
                      'errors': len(recorder.errors), 'report': str(args.out.resolve())}), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
