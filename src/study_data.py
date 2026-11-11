"""Prepare project-level ApacheJIT and untouched external JITMovingTarget tasks.

Run from the project root after importing bootstrap:
    python -c "import bootstrap; from src.study_data import main; main()"

No model is trained and no predictive test performance is inspected here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['la','ld','nf','nd','ns','ent','ndev','age','nuc','aexp','arexp','asexp']
APACHE_PROJECTS = [('camel','apache/camel'),('ignite','apache/ignite'),('flink','apache/flink')]
EXTERNAL_PERIODS = {'openstack':(1322599384,1393590700),'qt':(1308350292,1395090476)}
L2 = 1e-4


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def finite_float(value):
    try:
        result=float(value)
    except (TypeError,ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def timestamp_string(value):
    return datetime.fromtimestamp(int(value),timezone.utc).isoformat()


def external_rows(name, raw):
    start,end=EXTERNAL_PERIODS[name]
    accepted,excluded=[],[]
    reasons=Counter()
    missing_labels=0
    for row in raw:
        t=finite_float(row['author_date'])
        la,ld,nf=[finite_float(row[f]) for f in ['la','ld','nf']]
        if not np.all(np.isfinite([t,la,ld,nf])):
            reason='invalid_filter_fields'
        elif not start <= t < end:
            reason='outside_source_period'
        elif la <= 0:
            reason='no_lines_added'
        elif la+ld > 10000:
            reason='churn_above_10000'
        elif nf > 100:
            reason='files_above_100'
        else:
            reason=None
        if reason:
            reasons[reason]+=1
            excluded.append({'commit_id':row['commit_id'],'reason':reason})
            continue
        bug=finite_float(row['bugcount'])
        if not np.isfinite(bug):
            if str(row['bugcount']).strip().lower() not in {'','na','nan'}:
                raise ValueError(f'Unexpected nonnumeric bugcount in {name}: {row["bugcount"]}')
            bug=0.0
            missing_labels+=1
        if bug < 0:
            raise ValueError(f'Negative bugcount in {name}')
        copied=dict(row)
        copied['_label']=int(bug>0)
        copied['_timestamp']=int(t)
        accepted.append(copied)
    return accepted,excluded,dict(reasons),missing_labels


def temporal_split(rows, family):
    rows=sorted(rows,key=lambda row:(row['_timestamp'],row['commit_id']))
    timestamps=np.asarray([r['_timestamp'] for r in rows],dtype=np.int64)
    if family=='apache':
        before=np.flatnonzero(timestamps<1483228800)
        if not len(before):
            raise ValueError('No pre-2017 rows')
        target=int(math.floor(0.8*len(before)))
        cut=timestamps[before[target]]
        labels=np.where(timestamps>=1483228800,'test',np.where(timestamps>=cut,'val','train'))
        rule={'test_start_utc':'2017-01-01T00:00:00+00:00',
              'validation_fraction_of_pre2017':0.2,'train_val_cut_timestamp':int(cut),
              'rule':'per-project pre-2017 final 20% validation; 2017-2019 test'}
    else:
        cut1=timestamps[int(math.floor(0.6*len(rows)))]
        cut2=timestamps[int(math.floor(0.8*len(rows)))]
        labels=np.where(timestamps>=cut2,'test',np.where(timestamps>=cut1,'val','train'))
        rule={'train_val_cut_timestamp':int(cut1),'val_test_cut_timestamp':int(cut2),
              'target_fractions':[0.6,0.2,0.2],
              'rule':'chronological 60/20/20; equal cut-point timestamps go to later split'}
    return rows,labels,rule


def prepare_task(name,family,project,raw_path,source,rows,excluded,filter_audit):
    destination=ROOT/'data/study'/name
    destination.mkdir(parents=True,exist_ok=True)
    ids=[r['commit_id'] for r in rows]
    duplicates=[key for key,n in Counter(ids).items() if n>1]
    if duplicates:
        raise ValueError(f'Duplicate IDs in {name}: {duplicates[:5]}')
    rows,split,split_rule=temporal_split(rows,family)
    y=np.asarray([r['_label'] for r in rows],dtype=np.float64)
    times=np.asarray([r['_timestamp'] for r in rows],dtype=np.int64)
    ids=np.asarray([r['commit_id'] for r in rows],dtype='U64')
    X=np.asarray([[finite_float(r[f]) for f in FEATURES] for r in rows],dtype=np.float64)
    invalid_audit={}
    for j,f in enumerate(FEATURES):
        missing=~np.isfinite(X[:,j])
        negative=np.isfinite(X[:,j]) & (X[:,j]<0)
        invalid_audit[f]={'nonfinite_or_missing':int(missing.sum()),'negative':int(negative.sum())}
        X[negative,j]=np.nan
        if f!='ent':
            X[:,j]=np.log1p(X[:,j])
    train_mask=split=='train'
    medians=np.nanmedian(X[train_mask],axis=0)
    if not np.all(np.isfinite(medians)):
        raise ValueError(f'All-missing training feature in {name}')
    X=np.where(np.isfinite(X),X,medians[None,:])
    means=X[train_mask].mean(axis=0)
    scales=X[train_mask].std(axis=0,ddof=0)
    constant=scales<1e-12
    scales[constant]=1.0
    X=np.ascontiguousarray((X-means)/scales,dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError('Nonfinite preprocessed values')
    arrays={}
    splits={}
    for part in ['train','val','test']:
        mask=split==part
        n=int(mask.sum())
        positives=int(y[mask].sum())
        if not n or positives==0 or positives==n:
            raise ValueError(f'{name} {part} must contain both classes')
        arrays[f'X_{part}']=X[mask]
        arrays[f'y_{part}']=y[mask]
        arrays[f'{part}_ids']=ids[mask]
        arrays[f'timestamps_{part}']=times[mask]
        arrays[f'projects_{part}']=np.full(n,project,dtype=f'U{max(32,len(project))}')
        splits[part]={'n':n,'positive':positives,'negative':n-positives,
                      'positive_fraction':positives/n,
                      'date_min_utc':timestamp_string(times[mask].min()),
                      'date_max_utc':timestamp_string(times[mask].max()),
                      'min_timestamp':int(times[mask].min()),'max_timestamp':int(times[mask].max())}
    assert splits['train']['max_timestamp']<splits['val']['min_timestamp']
    assert splits['val']['max_timestamp']<splits['test']['min_timestamp']
    assert len(set(arrays['train_ids']) & set(arrays['val_ids']))==0
    assert len(set(arrays['train_ids']) & set(arrays['test_ids']))==0
    assert len(set(arrays['val_ids']) & set(arrays['test_ids']))==0
    n0=splits['train']['negative']; n1=splits['train']['positive']
    Xt=arrays['X_train']; yt=arrays['y_train']
    r=(Xt[yt==0].mean(axis=0)-Xt[yt==1].mean(axis=0))/4.0
    rnorm=float(np.linalg.norm(r))
    weights=np.where(yt==1,n0/n1,1.0)
    weights/=weights.sum()
    zero_grad_inf=abs(float(weights@(0.5-yt)))
    certificate={'lambda':L2,'r':r.tolist(),'r_norm':rnorm,
                 'lambda_minus_norm_r':L2-rnorm,'zero_grad_inf':zero_grad_inf,
                 'zero_objective':float(np.log(2.0)),
                 'derivation':'At all-zero parameters, the (one hidden-unit W column, corresponding v) principal Hessian block is [[lambda I, r],[r^T,lambda]], with r=(mean_class0_standardized_X-mean_class1_standardized_X)/4 under normalized balanced class weights. Its smallest eigenvalue is lambda-||r||; a negative value certifies a negative-curvature direction of the full actual training objective. Bias gradients vanish analytically because class weights balance the labels. This is a saddle-point/nonconvexity certificate, not a minimum certificate.'}
    metadata={'dataset':name,'family':family,'project':project,'features':FEATURES,
              'source':source,'raw_path':str(raw_path),'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              'retained_rows':len(rows),'filter_audit':filter_audit,
              'feature_audit':invalid_audit,'duplicate_commit_count':0,'split_rule':split_rule,'splits':splits,
              'preprocessing':{'fit_split':'train','log1p_features':[f for f in FEATURES if f!='ent'],
                               'medians_after_log1p':medians.tolist(),'means_after_imputation':means.tolist(),
                               'population_scales':scales.tolist(),'constant_features':[f for f,c in zip(FEATURES,constant) if c]},
              'positive_class_weight':n0/n1,'zero_point_curvature':certificate,
              'evaluation_role':'pilot-informed Apache extension' if family=='apache' else 'new external dataset; performance not inspected in preparation',
              'checks':{'finite_arrays':True,'disjoint_ids':True,'strict_time_order':True,'both_classes_each_split':True}}
    np.savez_compressed(destination/'dataset.npz',**arrays)
    write_json(destination/'metadata.json',metadata)
    with (destination/'split_assignments.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle); writer.writerow(['commit_id','project','author_date','split','label'])
        writer.writerows(zip(ids,[project]*len(ids),times,split,y.astype(int)))
    if excluded:
        with (destination/'excluded_rows.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['commit_id','reason']);writer.writeheader();writer.writerows(excluded)
    print(json.dumps({'dataset':name,'splits':{k:[v['n'],v['positive']] for k,v in splits.items()},
                      'lambda_minus_norm_r':L2-rnorm,'zero_grad_inf':zero_grad_inf}),flush=True)
    return {'dataset':name,'family':family,'path':str(destination.relative_to(ROOT)),
            'raw_sha256':metadata['raw_sha256'],'splits':splits,
            'lambda_minus_norm_r':L2-rnorm}


def main():
    root=ROOT/'data/study';root.mkdir(parents=True,exist_ok=True)
    apache_path=ROOT/'data/raw/apachejit_total.csv'
    all_apache=read_csv(apache_path)
    manifest=[]
    for name,project in APACHE_PROJECTS:
        selected=[]
        for row in all_apache:
            if row['project']!=project:
                continue
            r=dict(row)
            if r['buggy'] not in {'True','False'}:
                raise ValueError('Unexpected Apache label')
            r['_label']=int(r['buggy']=='True');r['_timestamp']=int(r['author_date'])
            selected.append(r)
        manifest.append(prepare_task(name,'apache',project,apache_path,
                         {'url':'https://raw.githubusercontent.com/hosseinkshvrz/apachejit/master/dataset/apachejit_total.csv',
                          'selection_rule':'Three largest non-Hadoop project identifiers by total record count'},
                         selected,[],{'raw_project_rows':len(selected),'excluded_rows':0}))
    for name in ['openstack','qt']:
        path=ROOT/'data/external_raw'/f'{name}.csv'
        raw=read_csv(path)
        rows,excluded,reasons,missing_labels=external_rows(name,raw)
        start,end=EXTERNAL_PERIODS[name]
        audit={'raw_rows':len(raw),'retained_rows':len(rows),'excluded_rows':len(excluded),
               'sequential_exclusion_reasons':reasons,'source_filter_start_timestamp':start,
               'source_filter_end_timestamp_exclusive':end,
               'source_filter_predicate':'start<=author_date<end AND la>0 AND la+ld<=10000 AND nf<=100',
               'blank_bugcount_labeled_negative':missing_labels,
               'label_policy':'bugcount>0; blank/NA bugcount maps to negative, matching source logical-NA-to-FALSE cleanup'}
        manifest.append(prepare_task(name,'external',name,path,
                      {'url':f'https://raw.githubusercontent.com/software-rebels/JITMovingTarget/master/data/{name}.csv',
                       'label_and_filter_source':'https://github.com/software-rebels/JITMovingTarget/blob/master/scripts/load_data.r',
                       'threshold_source':'https://github.com/software-rebels/JITMovingTarget/blob/master/scripts/experimental_settings.r',
                       'paper_doi':'10.1109/TSE.2017.2693980'},rows,excluded,audit))
    write_json(root/'manifest.json',{'datasets':manifest,'feature_order':FEATURES,
                 'preparation_only':True,'test_predictive_performance_computed':False,
                 'l2_for_curvature_certificate':L2,
                 'warning':'Apache extensions are pilot-informed; external tests require a frozen model-selection protocol before evaluation.'})


if __name__=='__main__':
    main()
