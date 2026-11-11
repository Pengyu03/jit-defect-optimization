"""Verify an actual negative-curvature saddle of each 32-unit training objective.

No training or held-out predictive evaluation is performed. Run from the root:
    python check_zero_saddle.py
"""
import bootstrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from src.model import TanhMLP


ROOT=Path(__file__).resolve().parent
DATASETS=['camel','ignite','flink','openstack','qt']


def main():
    records=[]
    for name in DATASETS:
        path=ROOT/'data/study'/name/'dataset.npz'
        data=np.load(path,allow_pickle=False)
        X,y=data['X_train'],data['y_train']
        model=TanhMLP(X,y,hidden=32,l2=1e-4)
        theta=np.zeros(model.size,dtype=np.float64)
        f0,g0=model.value_grad(theta)
        # Derive the mixed W/v Hessian block from the exact sample weights
        # used by the objective, rather than taking values from metadata.
        r_weighted=X.T@(model.weights*(0.5-y))
        r_classmeans=(X[y==0].mean(axis=0)-X[y==1].mean(axis=0))/4.0
        rnorm=float(np.linalg.norm(r_weighted))
        formula_error=float(np.max(np.abs(r_weighted-r_classmeans)))
        gradient_inf=float(np.linalg.norm(g0,ord=np.inf))
        if formula_error>1e-12 or gradient_inf>1e-12:
            raise AssertionError(f'{name}: balanced-weight zero-point identities failed')
        direction=np.zeros(model.size,dtype=np.float64)
        # W is row-major (d,h); use its first hidden-unit column.
        direction[np.arange(model.d)*model.h]=r_weighted/(rnorm*np.sqrt(2.0))
        first_v=model.d*model.h+model.h
        direction[first_v]=-1.0/np.sqrt(2.0)
        norm=float(np.linalg.norm(direction))
        analytic=model.l2-rnorm
        if analytic>=0 or abs(norm-1)>1e-12:
            raise AssertionError(f'{name}: no unit negative-curvature direction')
        probes=[]
        for epsilon in [1e-3,1e-4]:
            fp=model.fun(theta+epsilon*direction)
            fm=model.fun(theta-epsilon*direction)
            curvature=(fp-2*f0+fm)/(epsilon*epsilon)
            absolute_error=abs(curvature-analytic)
            if absolute_error>1e-5:
                raise AssertionError(f'{name}: finite-difference curvature mismatch {absolute_error}')
            if not (fp<f0 and fm<f0):
                raise AssertionError(f'{name}: perturbations did not strictly lower the objective')
            probes.append({'epsilon':epsilon,'f_plus':fp,'f_minus':fm,
                           'f_plus_minus_f_zero':fp-f0,'f_minus_minus_f_zero':fm-f0,
                           'central_difference_curvature':curvature,
                           'absolute_error_vs_analytic':absolute_error,
                           'both_perturbations_lower_loss':True})
        records.append({'dataset':name,'train_n':len(y),'hidden_units':32,
                        'parameters':model.size,'lambda':model.l2,'positive_weight':model.positive_weight,
                        'class_0_weight_sum':float(model.weights[y==0].sum()),
                        'class_1_weight_sum':float(model.weights[y==1].sum()),
                        'f_zero':f0,'zero_gradient_infinity_norm':gradient_inf,
                        'r_from_actual_objective_weights':r_weighted.tolist(),
                        'r_from_class_means':r_classmeans.tolist(),
                        'r_formula_max_absolute_error':formula_error,
                        'r_norm':rnorm,'principal_block_negative_eigenvalue':analytic,
                        'direction_norm':norm,'finite_difference_probes':probes,
                        'passed':True,
                        'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'all_passed':all(r['passed'] for r in records),
            'model_sha256':hashlib.sha256((ROOT/'src/model.py').read_bytes()).hexdigest(),
            'derivation':'At theta=0, sigma=1/2 and tanh=0. The only possibly nonzero gradient is output bias, whose balanced weighted residual sum is zero. For one hidden-unit W column and its output weight v, the principal Hessian block is [[lambda I,r],[r^T,lambda]], r=sum_i normalized_weight_i*(1/2-y_i)*x_i=(mean_class0_X-mean_class1_X)/4. Unit direction [r/||r||,-1]/sqrt(2) has curvature lambda-||r||. Negative curvature at this stationary point certifies the actual objective is nonconvex and theta=0 is not a local minimum.',
            'checks':'Recomputed from actual TanhMLP hidden=32; no training and no validation/test prediction.',
            'records':records}
    destination=ROOT/'research/zero_saddle_certificate.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'all_passed':report['all_passed'],'datasets':len(records),
                      'max_zero_gradient':max(r['zero_gradient_infinity_norm'] for r in records),
                      'max_curvature_error':max(p['absolute_error_vs_analytic'] for r in records for p in r['finite_difference_probes']),
                      'output':str(destination)},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
