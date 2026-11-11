"""Small reporting edge cases; no fitting or production result replay."""
import math
import json

import pytest

import analyze_study as analysis


def empty_failure_summary():
    return {'status':'numerical_failure', 'iterations':0,
            'eligible_for_tuning':False, 'selected_iteration':None,
            'final_loss':None, 'final_grad_inf':None}


def test_explicit_empty_failure_does_not_require_an_unwritten_history_csv(tmp_path):
    path=tmp_path/'history.csv'
    assert not path.exists()
    assert analysis.read_history(path,empty_failure_summary())==[]


@pytest.mark.parametrize('summary', [
    None,
    {'status':'evaluation_limit', 'iterations':10, 'eligible_for_tuning':True,
     'selected_iteration':5, 'final_loss':0.4, 'final_grad_inf':0.01},
    # A failure after the valid initial point still has a required history.
    {'status':'numerical_failure', 'iterations':0, 'eligible_for_tuning':False,
     'selected_iteration':0, 'final_loss':0.7, 'final_grad_inf':0.2},
    # Incomplete summaries must not silently authorize absent artifacts.
    {'status':'numerical_failure'},
])
def test_missing_history_is_rejected_without_explicit_empty_result(tmp_path,summary):
    with pytest.raises(FileNotFoundError):
        analysis.read_history(tmp_path/'history.csv',summary)


def test_all_missing_metric_displays_zero_available_denominator():
    entry={'n_runs':5, 'metrics':{'test_ap':analysis.stats([None]*5)}}
    assert analysis.tex_stat(entry,'test_ap')=='--- (0/5)'


def test_aggregate_retains_missing_run_and_reports_sample_sd_of_available_scores():
    rows=[]
    for seed,score in zip((10,11,12),(0.2,0.4,None)):
        rows.append({'dataset':'flink', 'hidden':32, 'method':'adam', 'seed':seed,
                     'status':'numerical_failure' if score is None else 'evaluation_limit',
                     'failure':score is None, 'test_available':score is not None,
                     'reached_final_tol_1e4':False, 'test_ap':score})
    result,=analysis.aggregate(rows)
    assert result['n_runs']==3
    assert result['failure_count']==1
    assert result['test_available_count']==2
    metric=result['metrics']['test_ap']
    assert metric['n_available']==2
    assert metric['mean']==pytest.approx(0.3)
    assert metric['sample_sd']==pytest.approx(math.sqrt(0.02))
    assert analysis.tex_stat(result,'test_ap').endswith(' (2/3)')


def test_tuning_review_respects_eligibility_and_ties_without_modifying_inputs(tmp_path):
    expected_changed={('camel','gd'),('ignite','momentum'),('ignite','adam'),
                      ('flink','gd'),('flink','adam'),('flink','lbfgs')}
    originals={}
    for dataset in analysis.APACHE:
        base=tmp_path/'tuning'/dataset
        base.mkdir(parents=True)
        screen={'protocol_hash':'frozen','methods':{}}
        selection={'protocol_hash':'frozen','methods':{}}
        for method in analysis.METHODS:
            screen['methods'][method]=[
                {'candidate':0,'eligible':False,'mean_validation_ap':0.95},
                {'candidate':2,'eligible':True,'mean_validation_ap':0.6},
                {'candidate':1,'eligible':True,'mean_validation_ap':0.6},
                {'candidate':3,'eligible':True,'mean_validation_ap':0.55}]
            selection['methods'][method]={'candidate':2 if (dataset,method) in expected_changed else 1,
                                         'mean_validation_ap':0.65}
        for filename,value in (('screen_rankings.json',screen),('selection.json',selection)):
            path=base/filename
            path.write_text(json.dumps(value),encoding='utf-8')
            originals[path]=path.read_bytes()
    review=analysis.tuning_review(tmp_path,'frozen')
    assert review['n_comparisons']==12
    assert review['changed_count']==6
    assert review['changed_fraction']==0.5
    assert all(row['screen_winner_candidate']==1 for row in review['comparisons'])
    assert {(row['dataset'],row['method']) for row in review['comparisons'] if row['changed']}==expected_changed
    assert len(review['source_files_sha256_relative_to_run'])==6
    assert all(path.read_bytes()==content for path,content in originals.items())
