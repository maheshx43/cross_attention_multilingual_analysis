"""CPU-only target-baseline summary on the exact saved E7 test panel and resamples."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'results_cache'
E4=CACHE/'E4_mask_sensitivity_fp32_v3'
E6=CACHE/'E6_fp32_regime_comparison_v1'
E7=CACHE/'E7_fp32_model_comparison_v1'
source=(E7/'e7_uncertainty.py').read_text()
tree=ast.parse(source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prediction_scores')
namespace={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),'saved_e7_prediction_scores','exec'),namespace)
scores=namespace['prediction_scores']
metrics=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='METRICS' for t in n.targets)))
support=json.loads((E6/'analysis/E6_primary_support.json').read_text())
config=json.loads((E7/'config.json').read_text())
old=pd.read_csv(E6/'analysis/E6_held_dyad_direction_predictions.csv')
saved=pd.read_csv(E7/'analysis/E7_held_out_scores.csv')
ids=support['test'];pairs=sorted(old.pair.unique());dyads=[p for p in pairs if p<'__'.join(reversed(p.split('__')))]
owner=np.array([dyads.index(min(p,'__'.join(reversed(p.split('__'))))) for p in pairs])
panels=[(s,r) for s in ['it2','nllb','madlad'] for r in ['TF','FG']]
data={};predictions={};points={}
for key in panels:
    study,regime=key
    path=E4/'collectors'/study/'all_rows.csv' if regime=='TF' else E6/'analysis'/study/'FG_with_LID.csv'
    frame=pd.read_csv(path,usecols=['sentence_id','pair','raw_lm'])
    y=frame.pivot(index='sentence_id',columns='pair',values='raw_lm').loc[ids,pairs].to_numpy(float)
    assert np.isfinite(y).all()
    data[key]=y;predictions[key]={};points[key]={}
    for model in ['both','target']:
        selected=old[(old.study==study)&(old.regime==regime)&(old.outcome=='raw_lm')&(old.model==model)]
        pred=selected.set_index('pair').loc[pairs].prediction_from_training.to_numpy(float)
        predictions[key][model]=pred;points[key][model]=scores(y,pred,np.ones((1,len(ids))))[0]
b=config['n_boot'];rng=np.random.default_rng(config['bootstrap_seed'])
draws={key:{model:np.empty((b,len(metrics))) for model in ['both','target']} for key in panels}
for begin in range(0,b,256):
    end=min(begin+256,b);count=end-begin
    meanings=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=count)
    edges=rng.multinomial(20,np.full(20,1/20),size=count)[:,owner]
    for key in panels:
        for model in ['both','target']:
            draws[key][model][begin:end]=scores(data[key],predictions[key][model],meanings,edges)
result=[]
for key in panels:
    study,regime=key
    for j,metric in enumerate(metrics):
        if metric!='direction_MSE':
            original=saved[(saved.study==study)&(saved.regime==regime)&(saved.model=='M1_additive')&
                (saved.metric==metric)&(saved.bootstrap=='meaning_and_dyad')].iloc[0]
            lo,hi=np.quantile(draws[key]['both'][:,j],[.025,.975])
            assert abs(points[key]['both'][j]-original.estimate)<1e-10
            assert np.allclose([lo,hi],[original.ci_low,original.ci_high],atol=1e-10,rtol=0), (key,metric)
        lo,hi=np.quantile(draws[key]['target'][:,j],[.025,.975])
        sign=1 if metric.endswith('R2') else -1
        delta=sign*(draws[key]['target'][:,j]-draws[key]['both'][:,j])
        dl,dh=np.quantile(delta,[.025,.975])
        result.append({'study':study,'regime':regime,'model':'target_only','metric':metric,
            'estimate':points[key]['target'][j],'ci_low':lo,'ci_high':hi,
            'improvement_vs_M1':sign*(points[key]['target'][j]-points[key]['both'][j]),
            'improvement_ci_low':dl,'improvement_ci_high':dh,'bootstrap':'same E7 paired meanings_and_dyads',
            'scope':'report-only secondary summary of saved E6 target predictions; no new fitting or GPU collection',
            'nominal_parameters':11})
pd.DataFrame(result).to_csv(ROOT/'E7_target_only_report_summary.csv',index=False)
(ROOT/'E7_target_only_report_validation.json').write_text(json.dumps({'saved_E7_M1_points_and_intervals_reproduced':True,
    'panels':6,'draws':b,'test_meanings':len(ids),'bootstrap_seed':config['bootstrap_seed'],
    'new_fitting':False,'new_GPU_collection':False,'new_result_scope':'target-only secondary baseline summarization for report',
    'numpy_version':np.__version__},indent=2)+'\n',encoding='utf-8')
print('All saved M1 points/intervals reproduce exactly; target-only baseline summarized with identical E7 draws.')
