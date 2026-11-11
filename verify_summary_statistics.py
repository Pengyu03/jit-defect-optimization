"""Independently audit the completed study's descriptive statistics using stdlib.

No fitting, scientific-library imports, prediction or checkpoint selection.
Usage: python verify_summary_statistics.py --run results/study_v2 \
    --out tests/summary_statistics_check.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
METHODS = ('gd', 'momentum', 'adam', 'lbfgs')
APACHE = ('camel', 'ignite', 'flink')
EXTERNAL = ('qt', 'openstack')
METRICS = ('test_ap', 'final_loss', 'elapsed_s', 'n_fun', 'n_grad')
THRESHOLDS = (1e-2, 1e-3, 1e-4)
CHANGE_TOLERANCE = 1e-12
FAILURES = {'numerical_failure', 'line_search_failed'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return None if value in (None, '') else float(value)


def run_key(row):
    return (row['experiment_stage'], row['dataset'], int(row['hidden']), row['method'], int(row['seed']))


def group_key(row):
    return (row['dataset'], int(row['hidden']), row['method'])


def folder_key(value):
    return str(value).replace('\\', '/')


def expected_runs():
    result = set()
    for dataset in APACHE:
        result.update(('main', dataset, 32, method, seed) for method in METHODS for seed in range(10, 15))
    for dataset in EXTERNAL:
        result.update(('external', dataset, 32, method, seed) for method in METHODS for seed in range(10, 13))
    result.update(('width', 'flink', hidden, method, seed)
                  for hidden in (8, 128) for method in ('adam', 'lbfgs') for seed in range(10, 13))
    return result


def descriptive(values):
    available = [number(value) for value in values if number(value) is not None]
    return {'n_available': len(available),
            'mean': statistics.mean(available) if available else None,
            'sample_sd': statistics.stdev(available) if len(available) > 1 else None}


class Audit:
    def __init__(self):
        self.errors = []
        self.check_count = 0
        self.files = {}
        self.maximum_absolute_error = 0.0

    def check(self, condition, label):
        self.check_count += 1
        if not condition:
            self.errors.append(label)

    def close(self, actual, saved, label):
        actual, saved = number(actual), number(saved)
        if actual is None or saved is None:
            self.check(actual is None and saved is None, label + ': missing-value agreement')
            return
        if not math.isfinite(actual) or not math.isfinite(saved):
            self.check(False, label + ': nonfinite value')
            return
        self.maximum_absolute_error = max(self.maximum_absolute_error, abs(actual - saved))
        self.check(math.isclose(actual, saved, rel_tol=1e-9, abs_tol=1e-10), label)

    def read(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if relative in self.files:
            self.check(self.files[relative] == digest, 'Input unchanged between reads: ' + relative)
        else:
            self.files[relative] = digest
        return payload.decode('utf-8-sig')

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path), newline='')))

    def finish(self):
        for relative, digest in self.files.items():
            path = ROOT / relative
            self.check(path.is_file() and sha256(path) == digest,
                       'Input unchanged at audit completion: ' + relative)


def audit_group_statistics(audit, rows, json_groups, csv_groups, expected, label):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    audit.check(set(grouped) == expected, label + ': expected group keys')
    maps = []
    for kind, entries in [('json', json_groups), ('csv', csv_groups)]:
        if entries is None:
            continue
        mapping = {group_key(entry): entry for entry in entries}
        audit.check(len(mapping) == len(entries) and set(mapping) == expected, label + ': unique ' + kind + ' groups')
        maps.append((kind, mapping))
    output = []
    for key, members in sorted(grouped.items()):
        statistics_by_metric = {metric: descriptive(row[metric] for row in members) for metric in METRICS}
        seeds = sorted(int(row['seed']) for row in members)
        for kind, mapping in maps:
            saved = mapping[key]
            audit.check(int(saved['n_runs']) == len(members), f'{label}/{key}: run denominator ({kind})')
            saved_seeds = saved['seeds'] if kind == 'json' else [int(value) for value in saved['seeds'].split(';')]
            audit.check(saved_seeds == seeds, f'{label}/{key}: seed identities ({kind})')
            for metric, result in statistics_by_metric.items():
                for stat, value in result.items():
                    reference = saved['metrics'][metric][stat] if kind == 'json' else saved[f'{metric}_{stat}']
                    audit.close(value, reference, f'{label}/{key}/{metric}/{stat} ({kind})')
        output.append({'dataset': key[0], 'hidden': key[1], 'method': key[2],
                       'n_runs': len(members), 'seeds': seeds, 'metrics': statistics_by_metric})
    return output


def direction(delta):
    return 'decrease' if delta < -CHANGE_TOLERANCE else 'increase' if delta > CHANGE_TOLERANCE else 'unchanged'


def change_summary(rows):
    changes = []
    for row in rows:
        df = float(row['final_loss']) - float(row['selected_loss'])
        da = float(row['test_final_ap']) - float(row['test_ap'])
        changes.append({'run': list(run_key(row)), 'folder': folder_key(row['folder']),
                        'loss_change': df, 'test_ap_change': da,
                        'category': 'loss_' + direction(df) + '_ap_' + direction(da)})
    scopes = {}
    for scope in ('all', 'main', 'external', 'width'):
        subset = changes if scope == 'all' else [row for row in changes if row['run'][0] == scope]
        scopes[scope] = {'n_runs': len(subset), 'counts': dict(sorted(Counter(row['category'] for row in subset).items()))}
    return {'delta': 'final minus validation-selected; paired loss and AP at each saved parameter point',
            'absolute_equality_tolerance': CHANGE_TOLERANCE, 'scopes': scopes, 'per_run': changes,
            'interpretation': 'Descriptive within-run continuation association, not an independent-sample test or causal estimate. The initial comparison point was selected using validation AP.'}


def audit_first_hits(audit, run_root, rows, saved_hits, saved_summaries):
    by_event = {(folder_key(row['folder']), float(row['threshold'])): row for row in saved_hits}
    expected_events = {(folder_key(row['folder']), threshold) for row in rows for threshold in THRESHOLDS}
    audit.check(len(by_event) == len(saved_hits) == 288 and set(by_event) == expected_events,
                'First-hit events cover all 96 runs and three thresholds exactly once')
    replayed = []
    for row in rows:
        folder = folder_key(row['folder'])
        history = audit.csv(run_root / folder / 'history.csv')
        for threshold in THRESHOLDS:
            hit = next((point for point in history if float(point['normalized_grad_inf']) <= threshold), None)
            saved = by_event[(folder, threshold)]
            reached = hit is not None
            observation = 'hit' if reached else 'failure_terminated' if row['status'] in FAILURES else 'right_censored'
            audit.check(saved['reached'] == str(reached), f'{folder}/{threshold}: first hit from recorded history')
            audit.check(saved['observation_type'] == observation and saved['termination_status'] == row['status'],
                        f'{folder}/{threshold}: event/censor/failure classification')
            audit.check(run_key(saved) == run_key(row), f'{folder}/{threshold}: event run identity')
            for metric in ('n_fun', 'n_grad', 'elapsed_s'):
                audit.close(hit[metric] if reached else None, saved['first_hit_' + metric],
                            f'{folder}/{threshold}: actual first-hit {metric}')
            audit.close(saved['observed_n_fun'], row['n_fun'], f'{folder}/{threshold}: observation count')
            audit.close(saved['observed_elapsed_s'], row['elapsed_s'], f'{folder}/{threshold}: observation time')
            replayed.append({'row': row, 'threshold': threshold, 'hit': hit, 'observation_type': observation})
    grouped = defaultdict(list)
    for event in replayed:
        grouped[group_key(event['row']) + (event['threshold'],)].append(event)
    saved_map = {(row['dataset'], int(row['hidden']), row['method'], float(row['threshold'])): row
                 for row in saved_summaries}
    audit.check(len(saved_map) == len(saved_summaries) == len(grouped) == 72 and set(saved_map) == set(grouped),
                'First-hit summary covers all 24 groups and three thresholds')
    for key, events in grouped.items():
        saved = saved_map[key]
        observed = [event['hit'] for event in events if event['hit'] is not None]
        audit.check(int(saved['n_runs']) == len(events) and int(saved['hits']) == len(observed), f'{key}: hit denominator')
        for kind in ('right_censored', 'failure_terminated'):
            audit.check(int(saved[kind]) == sum(event['observation_type'] == kind for event in events), f'{key}: {kind}')
        for metric in ('n_fun', 'n_grad', 'elapsed_s'):
            values = descriptive(point[metric] for point in observed)
            for stat in ('mean', 'sample_sd'):
                audit.close(values[stat], saved[f'first_hit_{metric}_{stat}_among_hits'], f'{key}: conditional {metric}/{stat}')
    scopes = {}
    for scope in ('all', 'main', 'external', 'width'):
        scopes[scope] = []
        for threshold in THRESHOLDS:
            events = [event for event in replayed if event['threshold'] == threshold
                      and (scope == 'all' or event['row']['experiment_stage'] == scope)]
            scopes[scope].append({'threshold': threshold, 'n_runs': len(events),
                'first_hits': sum(event['hit'] is not None for event in events),
                'final_points_meeting_threshold': sum(float(event['row']['final_normalized_grad_inf']) <= threshold for event in events),
                'observation_types': dict(Counter(event['observation_type'] for event in events))})
    h32 = []
    for method in METHODS:
        for threshold in THRESHOLDS:
            events = [event for event in replayed if event['threshold'] == threshold
                      and int(event['row']['hidden']) == 32 and event['row']['method'] == method]
            h32.append({'method': method, 'threshold': threshold, 'n_runs': len(events),
                'first_hits': sum(event['hit'] is not None for event in events),
                'final_points_meeting_threshold': sum(float(event['row']['final_normalized_grad_inf']) <= threshold for event in events)})
    return {'scopes': scopes, 'h32_first_vs_final': h32,
            'scope': 'First-hit costs are independently found in saved accepted-point histories; objective or gradient formulas are not recomputed here. First entry below a coarse threshold does not imply remaining there.'}


def verify(audit, run_root):
    out = run_root / 'analysis'
    protocol = audit.json(run_root / 'protocol.json')
    aggregate = audit.json(out / 'aggregate.json')
    rows = audit.csv(out / 'per_run.csv')
    groups_csv = audit.csv(out / 'aggregate.csv')
    width_csv = audit.csv(out / 'width_aggregate.csv')
    pairs = audit.csv(out / 'checkpoint_pairs.csv')
    hits = audit.csv(out / 'first_hits.csv')
    hit_summaries = audit.csv(out / 'first_hit_summary.csv')
    expected = expected_runs()
    keys = [run_key(row) for row in rows]
    audit.check(len(keys) == len(set(keys)) == 96 and set(keys) == expected, 'Exactly the 96 prespecified unique reporting fits')
    audit.check(aggregate['protocol_hash'] == protocol['protocol_hash'], 'Analysis and frozen protocol hashes agree')
    for row in rows:
        audit.check(row['protocol_hash'] == protocol['protocol_hash'], 'Per-run protocol: ' + row['folder'])
        for metric in METRICS + ('selected_loss', 'selected_grad_inf', 'test_final_ap', 'final_normalized_grad_inf'):
            audit.check(math.isfinite(float(row[metric])), 'Finite audited observation: ' + row['folder'] + '/' + metric)
    expected_groups = {(key[1], key[2], key[3]) for key in expected}
    audit.check(len(expected_groups) == 24, 'Plan contains 24 aggregation groups (20 main-width plus four added-width groups)')
    statistics_checked = audit_group_statistics(audit, rows, aggregate['groups'], groups_csv, expected_groups, 'all')
    width_rows = [row for row in rows if row['dataset'] == 'flink' and row['method'] in ('adam', 'lbfgs')
                  and int(row['hidden']) in (8, 32, 128) and int(row['seed']) in (10, 11, 12)]
    expected_width = {('flink', hidden, method) for hidden in (8, 32, 128) for method in ('adam', 'lbfgs')}
    audit.check(len(width_rows) == 18, 'Width comparison has 18 entries including six reused h32 fits')
    width_checked = audit_group_statistics(audit, width_rows, None, width_csv, expected_width, 'width matched seeds')
    by_folder = {folder_key(row['folder']): row for row in rows}
    pair_keys = [(folder_key(row['folder']), row['checkpoint']) for row in pairs]
    audit.check(len(pair_keys) == len(set(pair_keys)) == 192
                and set(pair_keys) == {(folder, kind) for folder in by_folder for kind in ('selected', 'final')},
                'Exactly 192 unique same-checkpoint pairs for the 96 fits')
    for pair in pairs:
        row = by_folder[folder_key(pair['folder'])]
        kind = pair['checkpoint']
        audit.check(group_key(pair) == group_key(row) and int(pair['seed']) == int(row['seed'])
                    and pair['status'] == row['status'], 'Checkpoint pair identity: ' + pair['folder'] + '/' + kind)
        for target, source in [('training_loss', kind + '_loss'), ('gradient_inf', kind + '_grad_inf'),
                               ('test_ap', 'test_ap' if kind == 'selected' else 'test_final_ap')]:
            audit.close(pair[target], row[source], 'Same-checkpoint correspondence: ' + pair['folder'] + '/' + kind + '/' + target)
    changes = change_summary(rows)
    observed_counts = changes['scopes']['all']['counts']
    claimed_counts = {'loss_decrease_ap_decrease': 85, 'loss_decrease_ap_increase': 3,
                      'loss_increase_ap_decrease': 1, 'loss_unchanged_ap_unchanged': 7}
    audit.check(observed_counts == claimed_counts, 'Reproduces the reported 85/96 loss-down/AP-down observation and all other categories')
    first_hits = audit_first_hits(audit, run_root, rows, hits, hit_summaries)
    audit.check([entry['first_hits'] for entry in first_hits['scopes']['all']] == [95, 64, 5],
                'Reproduces all-run first-hit counts 95/96, 64/96, 5/96')
    return {'protocol_hash': protocol['protocol_hash'], 'reporting_runs': len(rows),
            'aggregate_groups': len(statistics_checked), 'checkpoint_pairs': len(pairs),
            'aggregate_statistics_checked': statistics_checked, 'width_matched_statistics_checked': width_checked,
            'selected_to_final_changes': changes, 'first_hit_audit': first_hits,
            'termination_status_counts': dict(Counter(row['status'] for row in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'results/study_v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'tests/summary_statistics_check.json')
    args = parser.parse_args()
    audit = Audit()
    result = {}
    try:
        audit.read(Path(__file__))
        result = verify(audit, args.run.resolve())
        audit.finish()
    except Exception as exc:
        audit.errors.append(f'Audit exception: {type(exc).__name__}: {exc}')
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': not audit.errors,
              'check_count': audit.check_count, 'errors': audit.errors,
              'comparison_tolerances': {'absolute': 1e-10, 'relative': 1e-9,
                                        'change_classification_absolute': CHANGE_TOLERANCE},
              'maximum_absolute_comparison_error': audit.maximum_absolute_error,
              'audited_files_sha256': dict(sorted(audit.files.items())),
              'audited_paths_relative_to': 'project root; forward-slash separators',
              'scope': 'Independent standard-library aggregation, saved-checkpoint correspondence, and first-hit counting. No fitting, scientific-library imports or prediction. Seeds and width variants are not independent projects.',
              **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': audit.check_count, 'errors': len(audit.errors),
                      'aggregate_groups': result.get('aggregate_groups'),
                      'report': str(args.out.resolve())}, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
