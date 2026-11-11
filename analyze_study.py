"""Summarize a completed, separately evaluated multi-project optimization study.

No model is fit, no checkpoint is selected, and no test score is recomputed.
Run after study_runner.py evaluate:
    python analyze_study.py --run results/study_v2 --report-dir report
"""
from __future__ import annotations

import bootstrap
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parent
DATASETS=('camel','ignite','flink','openstack','qt')
APACHE=('camel','ignite','flink')
EXTERNAL=('openstack','qt')
METHODS=('gd','momentum','adam','lbfgs')
DISPLAY={'camel':'Camel','ignite':'Ignite','flink':'Flink','openstack':'OpenStack','qt':'Qt',
         'gd':'GD','momentum':'Momentum','adam':'Adam','lbfgs':'L-BFGS'}
SHORT={'gd':'GD','momentum':'Mom.','adam':'Adam','lbfgs':'L-BFGS'}
COLORS={'gd':'#2873B8','momentum':'#DD8932','adam':'#168A80','lbfgs':'#8B58A5'}
THRESHOLDS=(1e-2,1e-3,1e-4)
FAILURES={'numerical_failure','line_search_failed'}
NUMERIC_METRICS=(
    'elapsed_s','n_fun','n_grad','iterations','final_loss','final_grad_inf',
    'final_normalized_grad_inf','selected_loss','selected_grad_inf',
    'selected_normalized_grad_inf','selected_n_fun','selected_elapsed_s',
    'validation_ap_observed','test_ap','test_roc_auc','test_f1','test_log_loss',
    'test_final_ap','test_final_roc_auc','test_final_log_loss',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    if not rows:
        Path(path).write_text('',encoding='utf-8')
        return
    keys=list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def finite_number(value):
    return value is not None and np.isfinite(float(value))


def stats(values):
    usable=np.asarray([float(v) for v in values if finite_number(v)],dtype=float)
    return {'n_available':int(len(usable)),
            'mean':float(usable.mean()) if len(usable) else None,
            'sample_sd':float(usable.std(ddof=1)) if len(usable)>1 else None,
            'median':float(np.median(usable)) if len(usable) else None,
            'minimum':float(usable.min()) if len(usable) else None,
            'maximum':float(usable.max()) if len(usable) else None}


def read_history(path,summary=None):
    path=Path(path)
    # The runner omits CSV files when termination occurs before even the
    # initial point can be recorded. Do not treat an ordinary missing file as
    # such a failure: the summary must explicitly identify the empty result.
    empty_termination=(summary is not None
        and summary.get('status') in FAILURES|{'time_limit','evaluation_limit','iteration_limit'}
        and summary.get('iterations')==0
        and summary.get('eligible_for_tuning') is False
        and all(key in summary and summary[key] is None for key in
                ('selected_iteration','final_loss','final_grad_inf')))
    if not path.exists() and empty_termination:
        return []
    with path.open(encoding='utf-8',newline='') as handle:
        raw=list(csv.DictReader(handle))
    result=[]
    for row in raw:
        converted={key:float(value) for key,value in row.items() if value not in ('',None)}
        if 'normalized_grad_inf' not in converted and 'grad_inf' in converted:
            initial=raw[0].get('grad_inf',1.0)
            converted['normalized_grad_inf']=converted['grad_inf']/max(1.0,float(initial))
        result.append(converted)
    for key in ('n_fun','n_grad','elapsed_s'):
        values=np.asarray([r[key] for r in result])
        if len(values) and np.any(np.diff(values)<0):
            raise ValueError(f'Non-monotonic {key}: {path}')
    return result


def flatten_run(summary,history):
    keys=('dataset','method','seed','hidden','l2','experiment_stage','status','folder',
          'elapsed_s','n_fun','n_grad','iterations','final_loss','final_grad_inf',
          'final_normalized_grad_inf','selected_loss','selected_grad_inf',
          'selected_iteration','selected_n_fun','selected_elapsed_s','validation_ap_observed',
          'validation_threshold','run_hash','protocol_hash')
    row={key:summary.get(key) for key in keys}
    row['n_fun_budget']=summary['config']['max_evaluations']
    row['time_cap_s']=summary['config']['time_budget_s']
    row['failure']=summary['status'] in FAILURES
    row['reached_final_tol_1e4']=bool(finite_number(row['final_normalized_grad_inf'])
                                    and row['final_normalized_grad_inf']<=1e-4)
    scale=max(1.0,history[0]['grad_inf']) if history else None
    row['selected_normalized_grad_inf']=(row['selected_grad_inf']/scale
              if scale is not None and finite_number(row['selected_grad_inf']) else None)
    evaluation=summary.get('evaluation') or {}
    selected=evaluation.get('test') or {}
    final=evaluation.get('test_final') or {}
    row['test_available']=bool(selected)
    for key in ('ap','roc_auc','f1','log_loss'):
        row['test_'+key]=selected.get(key)
    for key in ('ap','roc_auc','log_loss'):
        row['test_final_'+key]=final.get(key)
    row['n_test']=evaluation.get('n_test')
    row['test_positive_fraction']=evaluation.get('test_positive_fraction')
    row['wolfe_failures']=summary.get('diagnostics',{}).get('wolfe_failures',0)
    row['armijo_fallbacks']=summary.get('diagnostics',{}).get('armijo_fallbacks',0)
    row['curvature_pairs_skipped']=summary.get('diagnostics',{}).get('curvature_pairs_skipped',0)
    return row


def first_hit_rows(row,history):
    output=[]
    for threshold in THRESHOLDS:
        hit=next((item for item in history
                  if finite_number(item.get('normalized_grad_inf'))
                  and item['normalized_grad_inf']<=threshold),None)
        output.append({'dataset':row['dataset'],'method':row['method'],'hidden':row['hidden'],
                       'seed':row['seed'],'experiment_stage':row['experiment_stage'],
                       'threshold':threshold,'reached':hit is not None,
                       'first_hit_n_fun':int(hit['n_fun']) if hit else None,
                       'first_hit_n_grad':int(hit['n_grad']) if hit else None,
                       'first_hit_elapsed_s':hit['elapsed_s'] if hit else None,
                       'observed_n_fun':row['n_fun'],'observed_elapsed_s':row['elapsed_s'],
                       'termination_status':row['status'],
                       'observation_type':'hit' if hit else ('failure_terminated' if row['failure'] else 'right_censored'),
                       'folder':row['folder']})
    return output


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['hidden'],row['method'])].append(row)
    output=[]
    for (dataset,hidden,method),items in sorted(groups.items(),key=lambda kv:(DATASETS.index(kv[0][0]),kv[0][1],METHODS.index(kv[0][2]))):
        result={'dataset':dataset,'hidden':hidden,'method':method,'n_runs':len(items),
                'seeds':sorted(r['seed'] for r in items),
                'status_counts':dict(Counter(r['status'] for r in items)),
                'failure_count':sum(r['failure'] for r in items),
                'test_available_count':sum(r['test_available'] for r in items),
                'final_tol_1e4_hits':sum(r['reached_final_tol_1e4'] for r in items),
                'final_tol_1e4_rate':sum(r['reached_final_tol_1e4'] for r in items)/len(items),
                'metrics':{metric:stats([r.get(metric) for r in items]) for metric in NUMERIC_METRICS}}
        output.append(result)
    return output


def aggregate_flat(entries):
    rows=[]
    for entry in entries:
        row={key:entry[key] for key in ('dataset','hidden','method','n_runs','failure_count',
                                       'test_available_count','final_tol_1e4_hits','final_tol_1e4_rate')}
        row['seeds']=';'.join(map(str,entry['seeds']))
        row['status_counts']=json.dumps(entry['status_counts'],sort_keys=True)
        for metric,values in entry['metrics'].items():
            for name,value in values.items():
                row[f'{metric}_{name}']=value
        rows.append(row)
    return rows


def group_map(entries):
    return {(e['dataset'],e['hidden'],e['method']):e for e in entries}


def width_subset(rows):
    return [row for row in rows if row['dataset']=='flink' and row['method'] in ('adam','lbfgs')
            and row['hidden'] in (8,32,128) and row['seed'] in (10,11,12)]


def latex_escape(value):
    return str(value).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')


def tex_stat(entry,metric,digits=3,bold=False):
    value=entry['metrics'][metric]
    if value['mean'] is None:
        return f"--- ({value['n_available']}/{entry['n_runs']})"
    content=f"{value['mean']:.{digits}f}"
    if value['sample_sd'] is not None:
        content+=r'\,\pm\,'+f"{value['sample_sd']:.{digits}f}"
    if bold:
        content=r'\mathbf{'+content+'}'
    text='$'+content+'$'
    if value['n_available']!=entry['n_runs']:
        text+=f" ({value['n_available']}/{entry['n_runs']})"
    return text


def table(path,columns,header,rows,caption,label,size=r'\small'):
    content=[r'\begin{table}[t]',r'\centering',size,
             r'\caption{'+caption+'}',r'\label{'+label+'}',
             r'\begin{tabular}{'+columns+'}',r'\toprule',
             ' & '.join(header)+r' \\',r'\midrule']
    content+=[' & '.join(map(str,row))+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    Path(path).write_text('\n'.join(content),encoding='utf-8')


def make_tables(report_dir,entries,rows,metadata,certificate,run,protocol):
    out=report_dir/'tables';out.mkdir(parents=True,exist_ok=True)
    gm=group_map(entries)
    data_rows=[]
    for dataset in DATASETS:
        md=metadata[dataset]
        cells=[DISPLAY[dataset]]
        for part in ('train','val','test'):
            s=md['splits'][part]
            cells.append(f"{s['n']:,} ({100*s['positive_fraction']:.1f}\\%)")
        data_rows.append(cells)
    table(out/'data_table.tex','lrrr',['Task','Training','Validation','Test'],data_rows,
          r'Chronological datasets: sample count (positive percentage). Each task has independently fitted training-only preprocessing. Apache tasks extend the inspected pilot; OpenStack and Qt are external tasks.',
          'tab:data')
    # Use scientific notation understood by ordinary LaTeX, without siunitx.
    def scientific(value):
        mantissa,exponent=f'{value:.1e}'.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'
    saddle_rows=[]
    for item in certificate['records']:
        error=max(p['absolute_error_vs_analytic'] for p in item['finite_difference_probes'])
        saddle_rows.append([DISPLAY[item['dataset']],f"{item['r_norm']:.4f}",
                           f"{item['principal_block_negative_eigenvalue']:.4f}",
                           scientific(item['zero_gradient_infinity_norm']),scientific(error)])
    table(out/'saddle_table.tex','lrrrr',
          ['Task',r'$\|r\|_2$',r'$\lambda-\|r\|_2$',r'$\|g(0)\|_\infty$','FD error'],saddle_rows,
          r'Nonconvexity certificate for the actual 32-unit objectives. The all-zero stationary point has a negative-curvature direction; central differences at $10^{-3}$ and $10^{-4}$ agree with the analytic curvature. FD error is the larger absolute error.',
          'tab:saddle',r'\footnotesize')
    ap_rows=[]
    for dataset in DATASETS:
        available=[gm.get((dataset,32,method)) for method in METHODS]
        values=[e['metrics']['test_ap']['mean'] for e in available if e and e['metrics']['test_ap']['mean'] is not None]
        best=max(values) if values else None
        ap_rows.append([DISPLAY[dataset]]+[
            tex_stat(e,'test_ap',3,bold=e['metrics']['test_ap']['mean']==best) if e else '---'
            for e in available])
    table(out/'main_ap_table.tex','lrrrr',['Task']+[DISPLAY[m] for m in METHODS],ap_rows,
          r'Test average precision at the validation-selected checkpoint, mean $\pm$ sample SD. Apache tasks use five paired initialization seeds; external tasks use three. Bold uses unrounded means and marks the largest within a task, not statistical significance. Missing-score denominators, if any, are shown explicitly.',
          'tab:main-ap',r'\footnotesize')
    optimization=[]
    for dataset in APACHE:
        for i,method in enumerate(METHODS):
            e=gm[(dataset,32,method)]
            calls=f"{e['metrics']['n_fun']['mean']:.0f} / {e['metrics']['n_grad']['mean']:.0f}"
            optimization.append([DISPLAY[dataset] if i==0 else '',DISPLAY[method],
              tex_stat(e,'final_loss',4),f"{e['final_tol_1e4_hits']}/{e['n_runs']}",
              calls,tex_stat(e,'elapsed_s',1)])
    table(out/'optimization_table.tex','llrcrr',
          ['Task','Method',r'$F_{\rm end}$',r'$R\leq10^{-4}$',r'$N_f / N_g$','Time (s)'],optimization,
          r'Apache optimization outcomes: loss/time are mean $\pm$ sample SD; call counts are rounded means. $R=\|g\|_\infty/\max(1,\|g_0\|_\infty)$; the hit column counts final iterates attaining the tolerance. $N_f$ includes loss-only and loss--gradient calls, while $N_g$ counts gradients; neither equalizes FLOPs. Unreached tolerance at resource stops is censored; failures are recorded separately. Neither is counted as convergence.',
          'tab:optimization',r'\scriptsize')
    wg=group_map(aggregate(width_subset(rows)))
    width_rows=[]
    for hidden in (8,32,128):
        for method in ('adam','lbfgs'):
            e=wg[('flink',hidden,method)]
            width_rows.append([str(hidden),DISPLAY[method],tex_stat(e,'final_loss',4),
                               tex_stat(e,'test_ap',3),f"{e['final_tol_1e4_hits']}/{e['n_runs']}",
                               tex_stat(e,'elapsed_s',1)])
    table(out/'width_table.tex','rlrrcr',
          ['Width','Method',r'$F_{\rm end}$','Test AP',r'$R\leq10^{-4}$','Time (s)'],width_rows,
          r'Flink width extension, mean $\pm$ sample SD, using matched seeds 10--12 at all widths, including the 32-unit subset of the main runs. Width-32 settings are reused, so this measures sensitivity under fixed settings. Training loss is from the final iterate; AP uses the validation-selected checkpoint.',
          'tab:width',r'\footnotesize')
    tuning_rows=[]
    for dataset in APACHE:
        selection=read_json(run/'tuning'/dataset/'selection.json')['methods']
        tuning_rows.append(selection_row(DISPLAY[dataset],selection))
    tuning_rows.append(selection_row('External (frozen)',read_json(run/'global_selection.json')['methods']))
    table(out/'tuning_table.tex','lrrrr',
          ['Scope',r'GD $\bar\alpha$',r'Momentum $(\alpha,\beta)$',r'Adam $\alpha$',r'L-BFGS $m$'],tuning_rows,
          r'Selected optimizer settings. Apache finalists use training/validation data only and two tuning seeds. External settings minimize mean Apache screening rank and are frozen before external predictive evaluation; no external hyperparameter search is performed.',
          'tab:tuning',r'\footnotesize')


def selection_row(scope,selection):
    settings={m:selection[m]['config'] for m in METHODS}
    return [scope,f"{settings['gd']['learning_rate']:g}",
            '$('+f"{settings['momentum']['learning_rate']:g}, {settings['momentum'].get('momentum',0.9):g}"+')$',
            f"{settings['adam']['learning_rate']:g}",str(settings['lbfgs']['history_size'])]


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
      'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
      'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,
      'axes.grid':True,'grid.color':'#E5E9EE','grid.linewidth':0.45,
      'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def observed_median(axis,items,histories,key,color,label,linestyle='-',individual=True):
    traces=[]
    for row in items:
        history=histories[row['folder']]
        x=np.asarray([h['n_fun'] for h in history])
        y=np.asarray([h[key] for h in history])
        valid=np.isfinite(x)&np.isfinite(y)
        if key=='normalized_grad_inf':
            valid&=y>0
        x,y=x[valid],y[valid]
        if not len(x):
            continue
        traces.append((x,y))
        if individual:
            axis.plot(x,y,color=color,alpha=.17,lw=.45,linestyle=linestyle)
    if not traces:
        return
    start=max(t[0][0] for t in traces);end=min(t[0][-1] for t in traces)
    if end<start:
        return
    grid=np.unique(np.r_[np.linspace(start,end,300),end])
    values=np.stack([y[np.maximum(0,np.searchsorted(x,grid,side='right')-1)] for x,y in traces])
    q1,med,q3=np.quantile(values,[.25,.5,.75],axis=0)
    axis.step(grid,med,where='post',color=color,lw=1.1,label=label,linestyle=linestyle)
    axis.fill_between(grid,q1,q3,step='post',color=color,alpha=.11,lw=0)


def plot_optimization(path,datasets,rows,histories,budget):
    fig,axes=plt.subplots(2,len(datasets),figsize=(6.5,3.65),squeeze=False)
    for col,dataset in enumerate(datasets):
        for method in METHODS:
            items=[r for r in rows if r['dataset']==dataset and r['method']==method and r['hidden']==32]
            observed_median(axes[0,col],items,histories,'loss',COLORS[method],DISPLAY[method])
            observed_median(axes[1,col],items,histories,'normalized_grad_inf',COLORS[method],DISPLAY[method])
        axes[0,col].set_title(DISPLAY[dataset]);axes[1,col].set_xlabel('Objective evaluations $N_f$')
        axes[1,col].set_yscale('log');axes[1,col].axhline(1e-4,color='#444444',lw=.7,ls=':')
        for ax in axes[:,col]:
            ax.set_xticks([0,budget/3,2*budget/3,budget])
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.set_xlim(left=0,right=budget)
        axes[0,col].tick_params(labelbottom=False)
    axes[0,0].set_ylabel('Training objective $F$')
    axes[1,0].set_ylabel('Stationarity residual $R$')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.98),ncol=4,frameon=False)
    fig.subplots_adjust(left=.09,right=.965,bottom=.11,top=.86,hspace=.12,wspace=.34)
    fig.savefig(path,dpi=300);plt.close(fig)


def plot_ap(path,rows,metadata):
    fig,axes=plt.subplots(1,5,figsize=(6.5,2.55),squeeze=False)
    for col,dataset in enumerate(DATASETS):
        ax=axes[0,col];scores=[]
        for position,method in enumerate(METHODS):
            items=sorted([r for r in rows if r['dataset']==dataset and r['hidden']==32 and r['method']==method],key=lambda r:r['seed'])
            values=np.asarray([r['test_ap'] for r in items if finite_number(r['test_ap'])])
            scores.extend(values.tolist())
            if len(values):
                offsets=np.linspace(-.12,.12,len(values))
                ax.scatter(position+offsets,values,s=10,color=COLORS[method],edgecolors='white',lw=.25,zorder=3)
                ax.plot([position-.22,position+.22],[values.mean()]*2,color='#111111',lw=1.1,zorder=4)
        baseline=metadata[dataset]['splits']['test']['positive_fraction']
        ax.axhline(baseline,color='#6B737C',ls=':',lw=.8)
        ax.set_title(DISPLAY[dataset]);ax.set_xticks(range(4),[SHORT[m] for m in METHODS],rotation=55,ha='right')
        low=min([baseline]+scores);high=max([baseline]+scores)
        pad=max(.025,(high-low)*.15)
        ax.set_ylim(max(0,low-pad),min(1,high+pad));ax.grid(axis='x',visible=False)
        ax.text(.03,.02,f'Base {baseline:.3f}',transform=ax.transAxes,fontsize=6,color='#616973')
    axes[0,0].set_ylabel('Test average precision')
    fig.subplots_adjust(left=.08,right=.965,bottom=.24,top=.91,wspace=.46)
    fig.savefig(path,dpi=300);plt.close(fig)


def plot_width(path,rows,histories,budget):
    items=width_subset(rows)
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.7))
    line_styles={8:'--',32:'-',128:':'}
    for hidden in (8,32,128):
        for method in ('adam','lbfgs'):
            group=[r for r in items if r['hidden']==hidden and r['method']==method]
            observed_median(axes[0],group,histories,'loss',COLORS[method],f'{DISPLAY[method]}, h={hidden}',line_styles[hidden],individual=False)
    axes[0].set(xlabel='Objective evaluations $N_f$',ylabel='Training objective $F$',xlim=(0,budget))
    axes[0].ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    axes[0].legend(fontsize=5.8,loc='upper right',ncol=2,frameon=False)
    for i,hidden in enumerate((8,32,128)):
        for shift,method in [(-.15,'adam'),(.15,'lbfgs')]:
            values=np.asarray([r['test_ap'] for r in items if r['hidden']==hidden and r['method']==method and finite_number(r['test_ap'])])
            if not len(values):
                continue
            axes[1].scatter(i+shift+np.linspace(-.045,.045,len(values)),values,s=15,color=COLORS[method],label=DISPLAY[method] if i==0 else None,zorder=3)
            axes[1].plot([i+shift-.09,i+shift+.09],[values.mean()]*2,color='#111111',lw=1.2)
    axes[1].set(xticks=range(3),xticklabels=['8','32','128'],xlabel='Hidden width',ylabel='Test average precision')
    axes[1].legend(frameon=False,fontsize=6.5);axes[1].grid(axis='x',visible=False)
    fig.subplots_adjust(left=.09,right=.985,top=.94,bottom=.18,wspace=.35)
    fig.savefig(path,dpi=300);plt.close(fig)


def first_hit_summary(hits):
    groups=defaultdict(list)
    for row in hits:
        groups[(row['dataset'],row['hidden'],row['method'],row['threshold'])].append(row)
    output=[]
    for key,items in sorted(groups.items()):
        observed=[r for r in items if r['reached']]
        row=dict(zip(('dataset','hidden','method','threshold'),key))
        row.update(n_runs=len(items),hits=len(observed),right_censored=sum(r['observation_type']=='right_censored' for r in items),
                   failure_terminated=sum(r['observation_type']=='failure_terminated' for r in items))
        for metric in ('first_hit_n_fun','first_hit_n_grad','first_hit_elapsed_s'):
            value=stats([r[metric] for r in observed])
            row[metric+'_mean_among_hits']=value['mean']
            row[metric+'_sample_sd_among_hits']=value['sample_sd']
        output.append(row)
    return output


def tuning_review(run,protocol_hash):
    """Describe the saved screening/finalist decisions without selecting again."""
    comparisons=[]
    inputs={}
    for dataset in APACHE:
        base=Path(run)/'tuning'/dataset
        screen_path,selection_path=base/'screen_rankings.json',base/'selection.json'
        screen,selection=read_json(screen_path),read_json(selection_path)
        if screen['protocol_hash']!=protocol_hash or selection['protocol_hash']!=protocol_hash:
            raise ValueError(f'Tuning review protocol mismatch: {dataset}')
        for path in (screen_path,selection_path):
            inputs[path.relative_to(run).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        for method in METHODS:
            eligible=[row for row in screen['methods'][method] if row['eligible']]
            if not eligible:
                raise ValueError(f'No eligible saved screening candidate: {dataset}/{method}')
            short=min(eligible,key=lambda row:(-row['mean_validation_ap'],row['candidate']))
            final=selection['methods'][method]
            comparisons.append({'dataset':dataset,'method':method,
                'screen_winner_candidate':short['candidate'],
                'screen_winner_mean_validation_ap':short['mean_validation_ap'],
                'finalist_winner_candidate':final['candidate'],
                'finalist_winner_mean_validation_ap':final['mean_validation_ap'],
                'changed':short['candidate']!=final['candidate']})
    changed=sum(row['changed'] for row in comparisons)
    return {'protocol_hash':protocol_hash,'n_comparisons':len(comparisons),
            'changed_count':changed,'changed_fraction':changed/len(comparisons),
            'source_files_sha256_relative_to_run':inputs,'comparisons':comparisons,
            'interpretation':'Descriptive comparison of the eligible short-screen winner and the saved full-budget finalist selection. AP values come from different budget stages; they are not test scores or unbiased gain estimates. No configuration is changed.'}


def write_notes(out,entries,rows,hits,protocol,completeness):
    lines=['# Multi-project optimization study: generated analysis','',
      f"Reporting fits: {completeness['observed_runs']}/{completeness['expected_runs']}; complete={completeness['complete']}.",'',
      'This analysis reads frozen training histories and separately saved test evaluations. It does not tune, refit, or select new checkpoints.','',
      '## Protocol and interpretation','',
      f"- Main objective-evaluation budget: {protocol['budget']['main_n_fun']}; wall-clock safety cap: {protocol['budget']['main_time_cap_s']} seconds.",
      '- n_fun includes loss-only and joint loss-gradient computations. These counts are not equal-cost FLOP units; n_grad and actual elapsed time are also reported.',
      '- The reported tolerance uses R = ||g||_inf / max(1, ||g_initial||_inf). Reaching R <= 1e-4 establishes approximate first-order stationarity only, not a minimum or global optimum.',
      '- Time/evaluation limits are retained as right-censoring for unreached tolerances. Failure terminations are listed separately; a stopping budget is never substituted as a successful hitting time.',
      '- Apache projects were chosen by sample size before viewing project-specific optimizer results, but follow the already inspected pooled pilot. They are pilot-informed extensions.',
      '- Qt/OpenStack optimizer settings were frozen from Apache validation-screen ranks. External validation chooses checkpoint and F1 threshold under the same fixed protocol; external test performance is evaluated afterward.',
      '- Mean +/- sample SD describes initialization variation, not confidence intervals. Seeds are not independent software projects. No p-values or significance claims are computed.',
      '- Training and predictive metrics must refer to the same checkpoint when studying their association. checkpoint_pairs.csv explicitly distinguishes final and validation-selected models.',
      '- Failed runs remain in per_run.csv and all denominators. Available last-accepted-point scores from failed runs remain labeled by failure status; missing metrics are counted explicitly rather than silently dropped.','',
      '## Main 32-unit results','',
      '| Dataset | Method | Runs | Test AP mean +/- SD | Final F mean | Final tolerance hits | Statuses |',
      '|---|---|---:|---:|---:|---:|---|']
    for e in entries:
        if e['hidden']!=32:
            continue
        value=e['metrics']['test_ap'];f=e['metrics']['final_loss']['mean']
        ap='missing' if value['mean'] is None else f"{value['mean']:.4f} +/- {value['sample_sd']:.4f}" if value['sample_sd'] is not None else f"{value['mean']:.4f}"
        if value['n_available']!=e['n_runs']:
            ap+=f" ({value['n_available']}/{e['n_runs']} available)"
        ftext='missing' if f is None else f'{f:.6f}'
        lines.append(f"| {DISPLAY[e['dataset']]} | {DISPLAY[e['method']]} | {e['n_runs']} | {ap} | {ftext} | {e['final_tol_1e4_hits']}/{e['n_runs']} | {e['status_counts']} |")
    lines+=['','## Output files','',
       '- per_run.csv: every reporting fit, including failures.',
       '- aggregate.csv / aggregate.json: per-dataset, width and optimizer descriptive statistics and availability counts.',
       '- first_hits.csv / first_hit_summary.csv: event indicators, actual first-hit costs, and censoring; conditional means are explicitly among hits only.',
       '- checkpoint_pairs.csv: same-checkpoint objective/AP pairs; exploratory, not causal evidence.',
       '- width_aggregate.csv: matched Flink seeds 10–12 at all widths.',
       '- tuning_review.json: saved short-screen versus finalist winner changes for all 12 Apache project/method combinations.',
       '- Four PNG figures and six LaTeX tables are copied to the report directory.','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'results/study_v2')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'report')
    args=parser.parse_args()
    run=args.run.resolve();out=(args.out or run/'analysis').resolve();report=args.report_dir.resolve()
    evaluation=read_json(run/'evaluation.json')
    training=read_json(run/'training_manifest.json')
    wrapper=read_json(run/'protocol.json');protocol=wrapper.get('protocol',wrapper)
    if not evaluation['complete'] or not training['complete']:
        raise ValueError('Complete all reporting fits and separate test evaluation before final analysis')
    summaries=evaluation['runs']
    keys=[(s['dataset'],s['hidden'],s['method'],s['seed']) for s in summaries]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate reporting task keys')
    if evaluation['protocol_hash']!=training['protocol_hash']:
        raise ValueError('Training and evaluation protocol hashes differ')
    if evaluation['protocol_hash']!=wrapper['protocol_hash']:
        raise ValueError('Saved protocol and evaluated protocol hashes differ')
    training_identities={s['folder']:s['run_hash'] for s in training['runs']}
    evaluated_identities={s['folder']:s['run_hash'] for s in summaries}
    if training_identities!=evaluated_identities:
        raise ValueError('Evaluated fits do not match the completed training manifest')
    if len(summaries)!=evaluation['expected_runs'] or len(summaries)!=evaluation['observed_runs']:
        raise ValueError('Evaluation counts disagree with its reporting records')
    histories={s['folder']:read_history(run/s['folder']/'history.csv',s) for s in summaries}
    rows=[flatten_run(s,histories[s['folder']]) for s in summaries]
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),r['hidden'],METHODS.index(r['method']),r['seed']))
    hits=[hit for row in rows for hit in first_hit_rows(row,histories[row['folder']])]
    entries=aggregate(rows)
    metadata={d:read_json(ROOT/'data/study'/d/'metadata.json') for d in DATASETS}
    certificate=read_json(ROOT/'research/zero_saddle_certificate.json')
    out.mkdir(parents=True,exist_ok=True)
    figures=report/'figures';figures.mkdir(parents=True,exist_ok=True)
    completeness={key:evaluation[key] for key in ('expected_runs','observed_runs','complete')}
    write_csv(out/'per_run.csv',rows);write_csv(out/'aggregate.csv',aggregate_flat(entries))
    write_csv(out/'first_hits.csv',hits);write_csv(out/'first_hit_summary.csv',first_hit_summary(hits))
    write_csv(out/'width_aggregate.csv',aggregate_flat(aggregate(width_subset(rows))))
    save_json(out/'tuning_review.json',tuning_review(run,evaluation['protocol_hash']))
    pairs=[]
    for row in rows:
        for kind in ('selected','final'):
            pairs.append({'dataset':row['dataset'],'method':row['method'],'hidden':row['hidden'],
              'seed':row['seed'],'checkpoint':kind,'training_loss':row[kind+'_loss'],
              'gradient_inf':row[kind+'_grad_inf'],
              'test_ap':row['test_ap'] if kind=='selected' else row['test_final_ap'],
              'status':row['status'],'folder':row['folder']})
    write_csv(out/'checkpoint_pairs.csv',pairs)
    save_json(out/'aggregate.json',{'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'protocol_hash':evaluation['protocol_hash'],'completeness':completeness,
      'dispersion':'sample SD across initialization seeds (not confidence intervals)',
      'test_checkpoint':'validation-selected; final point separately retained',
      'groups':entries})
    make_tables(report,entries,rows,metadata,certificate,run,protocol)
    style();budget=protocol['budget']['main_n_fun']
    plot_optimization(figures/'apache_optimization.png',APACHE,rows,histories,budget)
    plot_optimization(figures/'external_optimization.png',EXTERNAL,rows,histories,budget)
    plot_ap(figures/'test_ap.png',rows,metadata)
    plot_width(figures/'width_extension.png',rows,histories,budget)
    write_notes(out,entries,rows,hits,protocol,completeness)
    print(json.dumps({'analysis':str(out),'report_tables':str(report/'tables'),
                      'report_figures':str(figures),'completeness':completeness,
                      'reporting_failures':sum(r['failure'] for r in rows),
                      'tol_1e4_final_hits':sum(r['reached_final_tol_1e4'] for r in rows)},indent=2))


if __name__=='__main__':
    main()
