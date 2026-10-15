# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1
from pathlib import Path
import os
# Keep tiny linear algebra efficient; set before importing NumPy.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Local Jupyter runtime: edit the paths in the next cell.')


# %% original code cell 1; notebook cell 3
BASELINE_ROOT=Path('/content/drive/MyDrive/fp32_tag_intervention')
INPUTS={
    'it2':BASELINE_ROOT/'holdout'/'baseline_tf',
    'nllb':BASELINE_ROOT/'confirm'/'baseline_tf',
}
FROZEN_SPLIT=BASELINE_ROOT/'frozen_meaning_split.json'
REQUIRE_FROZEN_SPLIT=True
# Explicit input provenance: these must be the fp32 teacher-forced baseline rows.
FP32_BASELINE_CONFIRMED=True
N_BOOT=1000
SEED=20261002
OUTPUT_ROOT=BASELINE_ROOT/'row_level_variance'

# Optional: filtered existing 40-row fp32 / both-model OOF prediction CSVs.
# Accepted saved columns: pair (or direction), observed, predicted.
# Keep None to recompute exactly the same held-dyad protocol from these rows.
SAVED_LODO_PATHS={'it2':None,'nllb':None}

# Optional summary JSONs verify component means, never supply row values.
BASELINE_SUMMARIES={
    'it2':BASELINE_ROOT/'holdout'/'analysis'/'baseline_analysis.json',
    'nllb':BASELINE_ROOT/'confirm'/'analysis'/'baseline_analysis.json',
}
assert FP32_BASELINE_CONFIRMED, 'Confirm the original fp32 baseline provenance before running.'
assert N_BOOT>=1000, 'Use at least 1,000 whole-meaning bootstrap draws.'


# %% original code cell 2; notebook cell 5
"""Balanced-panel row-level variance decomposition; CPU only."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd

def read_panel(path, split_path=None, require_split=True):
    path=Path(path)
    files=sorted(path.glob('*.csv')) if path.is_dir() else [path]
    if not files or not all(p.is_file() for p in files):
        raise FileNotFoundError(f'No collector rows at {path}. Use the existing fp32 baseline_tf CSV folder; summary JSON cannot reconstruct row variance.')
    frames=[];hashes=[]
    for p in files:
        frames.append(pd.read_csv(p))
        hashes.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    d=pd.concat(frames,ignore_index=True)
    aliases={'meaning_id':['meaning_id','sentence_id'],'src':['src','source'],
             'tgt':['tgt','tag','target'],'LM':['LM','lm','lexical_mass','lexical']}
    for dest,candidates in aliases.items():
        if dest not in d:
            found=next((c for c in candidates if c in d),None)
            if found is None:raise ValueError(f'Missing {dest}; accepted names: {candidates}')
            d[dest]=d[found]
    assert d[['meaning_id','src','tgt','LM']].notna().all().all(),'Missing identity or LM'
    d['meaning_id']=d.meaning_id.astype(str);d['src']=d.src.astype(str);d['tgt']=d.tgt.astype(str)
    if 'text' in d:assert d.text.astype(str).eq(d.tgt).all(),'Tag/text mismatch: intervention rows are forbidden'
    if 'eligible' in d:
        assert d.eligible.astype(str).str.lower().isin(['true','1','1.0']).all(),'Ineligible rows; do not silently drop them'
    if 'dtype' in d:assert d.dtype.astype(str).str.contains('float32|fp32',case=False,regex=True).all()
    canonical=d.src+'__'+d.tgt
    if 'direction' in d:assert d.direction.astype(str).eq(canonical).all(),'direction must be src__tgt'
    if 'pair' in d:assert d.pair.astype(str).eq(canonical).all(),'pair identity mismatch'
    d['direction']=canonical
    assert not d.src.eq(d.tgt).any(),'Unexpected self-translation direction'
    assert not d.duplicated(['meaning_id','direction']).any(),'Duplicate meaning/direction cells'
    assert len(d)==18000 and d.meaning_id.nunique()==450 and d.direction.nunique()==40,(
        'Expected exactly 18,000 rows, 450 meanings and 40 directions',len(d),d.meaning_id.nunique(),d.direction.nunique())
    assert d.groupby('meaning_id').size().eq(40).all()
    assert d.groupby('direction').size().eq(450).all()
    assert d.src.nunique()==11 and d.tgt.nunique()==11 and set(d.src)==set(d.tgt)
    dirs=set(d.direction)
    assert all(b+'__'+a in dirs for a,b in (p.split('__') for p in dirs)),'Directions must form 20 reciprocal dyads'
    d['LM']=pd.to_numeric(d.LM,errors='raise')
    assert np.isfinite(d.LM).all() and d.LM.between(0,1).all(),'LM must be finite mass, not percentages'
    if 'lexical' in d:assert np.allclose(d.LM,d.lexical,atol=1e-7,rtol=0),'LM differs from lexical component'
    if 'corpus_sentence_id' in d:assert d.groupby('meaning_id').corpus_sentence_id.nunique().eq(1).all()
    split_hash=None
    if split_path is not None and Path(split_path).is_file():
        raw=Path(split_path).read_bytes();split=json.loads(raw)
        assert set(d.meaning_id)==set(map(str,split['baseline450'])),'Rows differ from frozen baseline450 split'
        split_hash=hashlib.sha256(raw).hexdigest()
    elif require_split:
        raise FileNotFoundError(f'Frozen split missing: {split_path}. Supply it, or explicitly set REQUIRE_FROZEN_SPLIT=False for a separately verified baseline export.')
    return d,{'input_files':hashes,'rows':len(d),'meanings':450,'directions':40,
              'frozen_split_sha256':split_hash,'split_membership_verified':split_hash is not None}

def design(g,kind):
    langs=sorted(set(g.src)|set(g.tgt));parts=[np.ones((len(g),1))]
    for col in (['src','tgt'] if kind=='both' else [kind]):
        block=np.column_stack([g[col].eq(l).to_numpy(float) for l in langs])
        parts.append(block-1/len(langs))
    return np.column_stack(parts)

def operators(g):
    X=design(g,'both');assert np.linalg.matrix_rank(X)==21
    P={k:design(g,k)@np.linalg.pinv(design(g,k)) for k in ['src','tgt','both']}
    H=np.zeros((len(g),len(g)));seen=set();folds=[]
    for i,row in g.iterrows():
        if i in seen:continue
        j=g.index[(g.src==row.tgt)&(g.tgt==row.src)].item();test=np.array([i,j]);seen.update(test)
        train=np.ones(len(g),bool);train[test]=False
        assert np.linalg.matrix_rank(X[train])==21,'Held dyad makes additive design unidentifiable'
        H[np.ix_(test,np.flatnonzero(train))]=X[test]@np.linalg.pinv(X[train])
        folds.append({'fold':len(folds),'held_directions':g.direction.iloc[test].tolist(),'training_rank':21})
    assert len(folds)==20
    return P,H,folds

def panel_matrix(d,column):
    y=d.pivot(index='meaning_id',columns='direction',values=column).sort_index().sort_index(axis=1)
    assert y.notna().all().all(),'Incomplete crossed panel'
    return y.index.to_list(),y.columns.to_list(),y.to_numpy(float)

def variance_statistics(Y,P,H,weights):
    # Exact OLS for a balanced crossed panel, using sufficient statistics.
    # Each weight resamples an entire meaning's 40-direction block.
    M,D=Y.shape;W=np.asarray(weights,float);n=W.sum(axis=1)
    means=(W@Y)/n[:,None];mu=means.mean(axis=1)
    second=W@np.sum(Y*Y,axis=1)
    total=second-n*D*mu**2
    assert (total>0).all(),'Zero total variance'
    ss_dir=n*np.sum((means-mu[:,None])**2,axis=1)
    mbar=Y.mean(axis=1)
    ss_meaning=D*(W@(mbar*mbar)-n*mu**2)
    ss_res=total-ss_dir-ss_meaning
    assert np.min(ss_res)>-1e-9,'Numerical variance partition failed'
    within=total-ss_dir
    vals={'direction_share':ss_dir/total,'meaning_share':ss_meaning/total,
          'residual_share':ss_res/total,'R2_full_direction':ss_dir/total,
          'R2_full_direction_plus_meaning':(ss_dir+ss_meaning)/total,
          'within_direction_SD':np.sqrt(np.maximum(within,0)/(n*D)),
          'between_direction_SD':np.sqrt(ss_dir/(n*D)),
          'total_row_SD':np.sqrt(total/(n*D))}
    for k,label in [('tgt','target'),('src','source'),('both','source_plus_target')]:
        fitted=means@P[k].T
        miss=n*np.sum((means-fitted)**2,axis=1)
        vals['R2_'+label]=1-(within+miss)/total
        if k=='both':
            vals['R2_source_plus_target_plus_meaning']=1-(within+miss-ss_meaning)/total
            vals['direction_specific_gap']=miss/total
    pred=means@H.T
    held_error=np.sum((means-pred)**2,axis=1)
    vals['held_dyad_row_R2']=1-(within+n*held_error)/total
    vals['held_dyad_direction_mean_R2']=1-held_error/np.sum((means-mu[:,None])**2,axis=1)
    vals['meaning_gain_after_additive']=ss_meaning/total
    vals['non_language_share']=within/total
    vals['non_nested_R2_difference']=vals['R2_source_plus_target_plus_meaning']-vals['R2_full_direction']
    assert np.allclose(vals['direction_share']+vals['meaning_share']+vals['residual_share'],1,atol=1e-9)
    return vals,{'total':total,'direction':ss_dir,'meaning':ss_meaning,'residual':ss_res}

def length_checks(d,Y):
    cols=[]
    for options in [['source_len','src_len','source_token_length'],['target_len','tgt_len','target_token_length']]:
        col=next((c for c in options if c in d),None)
        if col:cols.append(col)
    if len(cols)!=2:return {'status':'skipped: source and target token-length columns both required'}
    X=np.stack([panel_matrix(d,c)[2] for c in cols],axis=-1)
    assert np.isfinite(X).all() and (X>=0).all()
    # Raw token lengths, without transformations. Partial out direction, then direction+meaning.
    total=np.sum((Y-Y.mean())**2);yd=Y-Y.mean(axis=0,keepdims=True)
    xd=X-X.mean(axis=0,keepdims=True)
    coef=np.linalg.lstsq(xd.reshape(-1,2),yd.ravel(),rcond=None)[0]
    sse_dl=np.sum((yd-np.einsum('mdk,k->md',xd,coef))**2)
    ym=yd-yd.mean(axis=1,keepdims=True)
    xm=xd-xd.mean(axis=1,keepdims=True)
    coef2=np.linalg.lstsq(xm.reshape(-1,2),ym.ravel(),rcond=None)[0]
    sse_dml=np.sum((ym-np.einsum('mdk,k->md',xm,coef2))**2)
    return {'status':'point estimates only; exploratory, no bootstrap CI','columns':cols,
        'R2_direction_plus_lengths':float(1-sse_dl/total),
        'R2_direction_plus_meaning_plus_lengths':float(1-sse_dml/total),
        'length_gain_after_direction':float((np.sum(yd*yd)-sse_dl)/total),
        'meaning_gain_after_direction_and_lengths':float((sse_dl-sse_dml)/total),
        'length_gain_after_direction_and_meaning':float((np.sum(ym*ym)-sse_dml)/total),
        'interpretation':'Length gain after direction is not exclusively a meaning effect: lengths can also vary by direction within meaning. Meaning gain conditional on lengths is a nested comparison, not a causal fraction.'}

def analyze_panel(d,n_boot=1000,seed=20261002,saved_predictions=None):
    ids,dirs,Y=panel_matrix(d,'LM')
    g=d[['direction','src','tgt']].drop_duplicates().set_index('direction').loc[dirs].reset_index()
    P,H,folds=operators(g)
    rng=np.random.default_rng(seed)
    W=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n_boot)
    point,ss=variance_statistics(Y,P,H,np.ones((1,len(ids))))
    boot,_=variance_statistics(Y,P,H,W)
    records=[]
    for metric,value in point.items():
        lo,hi=np.quantile(boot[metric],[.025,.975])
        records.append({'component':'LM','metric':metric,'estimate':float(value[0]),'ci_low':float(lo),'ci_high':float(hi)})
    for column in ['eos','punctuation']:
        if column not in d:continue
        _,_,C=panel_matrix(d,column)
        assert np.isfinite(C).all() and (C>=0).all() and (C<=1).all()
        pt,_=variance_statistics(C,P,H,np.ones((1,len(ids))));bt,_=variance_statistics(C,P,H,W)
        for metric in ['direction_share','meaning_share','residual_share','R2_source_plus_target','R2_full_direction','held_dyad_row_R2']:
            lo,hi=np.quantile(bt[metric],[.025,.975])
            records.append({'component':column,'metric':metric,'estimate':float(pt[metric][0]),'ci_low':float(lo),'ci_high':float(hi)})
    observed=Y.mean(axis=0);pred=H@observed
    verification='Recomputed from the collector rows using the existing 20-fold leave-reciprocal-dyad-out protocol.'
    if saved_predictions is not None:
        saved=pd.read_csv(saved_predictions)
        if 'model' in saved:saved=saved[saved.model.isin(['both','source_plus_target'])]
        if 'mode' in saved and saved['mode'].nunique()>1:
            raise ValueError('Saved OOF CSV has several modes. Export only the fp32 baseline450 / both model rows before using it.')
        key='direction' if 'direction' in saved else 'pair'
        assert not saved.duplicated(key).any()
        saved=saved.set_index(key).loc[dirs]
        assert np.allclose(saved.observed,observed,atol=1e-9,rtol=0),'Saved OOF rows do not match this baseline'
        assert np.allclose(saved.predicted,pred,atol=1e-9,rtol=0),'Saved OOF predictions differ from this protocol'
        pred=saved.predicted.to_numpy(float)
        verification='Supplied existing OOF predictions verified and used; all values match recomputation.'
    predictions=pd.DataFrame({'direction':dirs,'observed_mean_LM':observed,'held_dyad_predicted_mean_LM':pred})
    dyad_map={p:f['fold'] for f in folds for p in f['held_directions']}
    predictions['fold']=predictions.direction.map(dyad_map)
    row_predictions=d[['meaning_id','src','tgt','direction','LM']].merge(predictions[['direction','held_dyad_predicted_mean_LM','fold']],on='direction',validate='many_to_one')
    assert np.isclose(1-np.sum((row_predictions.LM-row_predictions.held_dyad_predicted_mean_LM)**2)/np.sum((row_predictions.LM-row_predictions.LM.mean())**2),point['held_dyad_row_R2'][0])
    return pd.DataFrame(records),predictions,row_predictions,{
        'sum_of_squares':{k:float(v[0]) for k,v in ss.items()},'degrees_of_freedom':{'total':17999,'direction':39,'meaning':449,'residual':17511},
        'mean_LM':float(Y.mean()),'n_boot':n_boot,'bootstrap_seed':seed,'folds':folds,
        'OOF_prediction_provenance':verification,'length_checks':length_checks(d,Y),
        'scope':'Balanced fixed-effect variance shares, not random-effects variance components. Residual combines direction-by-meaning interaction and measurement/numerical noise. Meanings remain seen across training directions in the dyad holdout; this is not joint unseen-meaning generalization.',
        'bootstrap':'Whole-meaning pairs bootstrap; every replicate refits direction, additive, meaning and held-dyad predictions. The 40 directions and observed graph are fixed. Percentile 95% intervals; no decoder-level or training uncertainty.',
        'non_nested_gap_warning':'R2(source+target+meaning) minus R2(direction) is meaning share minus the additive misspecification share. It is not a clean sentence-level variance estimate. Use meaning_share and non_language_share, or add meaning to full direction.'}

def self_test():
    # Independent dense OLS checks: actual 11-language/20-dyad graph, 12 synthetic meanings.
    names=['asm_Beng','ben_Beng','guj_Gujr','hin_Deva','kan_Knda','mal_Mlym','mar_Deva','ory_Orya','pan_Guru','tam_Taml','tel_Telu']
    edges=[(0,1),(3,7),(6,2),(2,8),(1,6),(4,9),(9,5),(9,10),(10,5),(4,5),(1,4),(3,4),(3,5),(3,9),(7,10),(0,4),(1,9),(2,5),(6,4),(8,9)]
    pairs=sorted([(names[a],names[b]) for a,b in edges]+[(names[b],names[a]) for a,b in edges])
    g=pd.DataFrame(pairs,columns=['src','tgt']);g['direction']=g.src+'__'+g.tgt
    P,H,_=operators(g);rng=np.random.default_rng(14);M=12;D=40
    X=design(g,'both');Y=.3+(X@rng.normal(0,.015,X.shape[1]))[None,:]+rng.normal(0,.02,(M,1))+rng.normal(0,.01,(M,D))
    vals,ss=variance_statistics(Y,P,H,np.ones((1,M)))
    total=np.sum((Y-Y.mean())**2)
    for kind,label in [('src','source'),('tgt','target'),('both','source_plus_target')]:
        Z=np.tile(design(g,kind),(M,1));fit=Z@np.linalg.lstsq(Z,Y.ravel(),rcond=None)[0]
        assert np.isclose(vals['R2_'+label][0],1-np.sum((Y.ravel()-fit)**2)/total,atol=1e-12)
    Md=np.repeat(np.eye(M),D,axis=0);Dd=np.tile(np.eye(D),(M,1))
    Z=np.column_stack([Md,Dd]);fit=Z@np.linalg.lstsq(Z,Y.ravel(),rcond=None)[0]
    assert np.isclose(vals['R2_full_direction_plus_meaning'][0],1-np.sum((Y.ravel()-fit)**2)/total,atol=1e-12)
    Z=np.column_stack([Md,np.tile(X,(M,1))]);fit=Z@np.linalg.lstsq(Z,Y.ravel(),rcond=None)[0]
    assert np.isclose(vals['R2_source_plus_target_plus_meaning'][0],1-np.sum((Y.ravel()-fit)**2)/total,atol=1e-12)
    w=rng.multinomial(M,np.full(M,1/M));v,_=variance_statistics(Y,P,H,w[None,:]);expanded=Y[np.repeat(np.arange(M),w)]
    vv,_=variance_statistics(expanded,P,H,np.ones((1,M)))
    for key in v:assert np.allclose(v[key],vv[key],atol=1e-11),key
    assert np.allclose(ss['total'],ss['direction']+ss['meaning']+ss['residual'])
    return 'Passed: dense OLS equivalence, variance closure, held-dyad design ranks, and weighted/explicit bootstrap equivalence.'

ANALYSIS_SOURCE_SHA256='e924a552e28930ed67d0b9e1e00edfea3590b4f7b59b8411e9f22f3ae79da403'
print(self_test())


# %% original code cell 3; notebook cell 7
from datetime import datetime,timezone
from IPython.display import display
import platform
assert set(INPUTS)=={'it2','nllb'}
RUN_DIR=OUTPUT_ROOT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR.mkdir(parents=True,exist_ok=False)
ALL_RESULTS=[];PANELS={};AUDITS={};ROW_PREDICTIONS={}
started=time.perf_counter()
for model,path in INPUTS.items():
    t0=time.perf_counter()
    d,input_audit=read_panel(path,FROZEN_SPLIT,REQUIRE_FROZEN_SPLIT)
    PANELS[model]=d
    summary_path=BASELINE_SUMMARIES.get(model)
    if summary_path and Path(summary_path).is_file():
        raw=Path(summary_path).read_bytes();baseline=json.loads(raw)
        assert baseline.get('kind')==model, 'Baseline summary is for the other model.'
        for column,expected in baseline['tf_component_means'].items():
            if column in d:
                assert abs(float(d[column].mean())-expected)<1e-7,(model,column,'component mean mismatch')
        input_audit['baseline_summary_sha256']=hashlib.sha256(raw).hexdigest()
        input_audit['component_means_match_summary']=True
    results,direction_predictions,row_predictions,analysis_audit=analyze_panel(
        d,n_boot=N_BOOT,seed=SEED,saved_predictions=SAVED_LODO_PATHS[model])
    results.insert(0,'model',model);ALL_RESULTS.append(results)
    ROW_PREDICTIONS[model]=row_predictions
    results.to_csv(RUN_DIR/f'{model}_all_metrics.csv',index=False)
    direction_predictions.to_csv(RUN_DIR/f'{model}_held_dyad_direction_predictions.csv',index=False)
    row_predictions.to_csv(RUN_DIR/f'{model}_held_dyad_row_predictions.csv',index=False)
    AUDITS[model]={'input':input_audit,'analysis':analysis_audit,'elapsed_seconds':time.perf_counter()-t0}
    print(f'{model}: validated {len(d):,} rows; completed {N_BOOT:,} bootstrap draws in {AUDITS[model]["elapsed_seconds"]:.1f}s including reads/writes.')

assert set(PANELS['it2'].meaning_id)==set(PANELS['nllb'].meaning_id), 'Models use different meaning IDs.'
assert set(PANELS['it2'].direction)==set(PANELS['nllb'].direction), 'Models use different direction graphs.'
if all('corpus_sentence_id' in d for d in PANELS.values()):
    a=PANELS['it2'].groupby('meaning_id').corpus_sentence_id.first().astype(str).sort_index()
    b=PANELS['nllb'].groupby('meaning_id').corpus_sentence_id.first().astype(str).sort_index()
    assert a.equals(b), 'Meaning IDs map to different corpus sentences across models.'
ALL_RESULTS=pd.concat(ALL_RESULTS,ignore_index=True)
ALL_RESULTS.to_csv(RUN_DIR/'all_models_all_metrics.csv',index=False)
manifest={'run_utc':RUN_DIR.name,'n_boot':N_BOOT,'seed':SEED,'models':AUDITS,
    'precision_attestation':'Original fp32 teacher-forced baseline, confirmed in configuration.',
    'numpy_version':np.__version__,'pandas_version':pd.__version__,'python_version':platform.python_version(),
    'engine_sha256':ANALYSIS_SOURCE_SHA256,
    'cross_model_meaning_and_direction_checks':'passed','elapsed_seconds':time.perf_counter()-started}
(RUN_DIR/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Results:',RUN_DIR)


# %% original code cell 4; notebook cell 9
MAIN_METRICS=[
    ('direction_share','Direction share of total row variance'),
    ('meaning_share','Meaning share of total row variance'),
    ('residual_share','Residual share of total row variance'),
    ('R2_target','Target-only row R² (in sample)'),
    ('R2_source','Source-only row R² (in sample)'),
    ('R2_source_plus_target','Source + target row R² (in sample)'),
    ('R2_full_direction','Full-direction row R² (in sample)'),
    ('R2_source_plus_target_plus_meaning','Source + target + meaning row R² (in sample)'),
    ('R2_full_direction_plus_meaning','Full direction + meaning row R² (in sample)'),
    ('held_dyad_row_R2','Held-dyad row-level R² (out of fold)'),
    ('held_dyad_direction_mean_R2','Held-dyad direction-mean R² (out of fold)'),
]
lm=ALL_RESULTS[ALL_RESULTS.component=='LM']
def entry(model,metric):
    return lm[(lm.model==model)&(lm.metric==metric)].iloc[0]
rows=[]
for metric,label in MAIN_METRICS:
    row={'Metric':label}
    for model in INPUTS:
        v=entry(model,metric)
        row[model]=f'{100*v.estimate:.2f}% [{100*v.ci_low:.2f}, {100*v.ci_high:.2f}]' if metric.endswith('_share') else f'{v.estimate:.4f} [{v.ci_low:.4f}, {v.ci_high:.4f}]'
    rows.append(row)
REPORT_TABLE=pd.DataFrame(rows)
display(REPORT_TABLE)
REPORT_TABLE.to_csv(RUN_DIR/'report_table.csv',index=False)
notes=[]
for model in INPUTS:
    direction=entry(model,'direction_share');held=entry(model,'held_dyad_row_R2');meanheld=entry(model,'held_dyad_direction_mean_R2')
    sentence=(f'{model.upper()}: language pair explains {100*direction.estimate:.2f}% '
        f'[{100*direction.ci_low:.2f}, {100*direction.ci_high:.2f}] of sentence-level lexical-mass variance in sample; '
        f'held-dyad additive prediction explains {100*held.estimate:.2f}% '
        f'[{100*held.ci_low:.2f}, {100*held.ci_high:.2f}] of row-level variance, '
        f'versus {100*meanheld.estimate:.2f}% of variance in direction means under the same held-dyad protocol.')
    notes.append(sentence);print(sentence)
    print('Sums of squares:',AUDITS[model]['analysis']['sum_of_squares'])
    print('Length checks:',json.dumps(AUDITS[model]['analysis']['length_checks'],indent=2))
gaps=lm[lm.metric.isin(['direction_specific_gap','meaning_gain_after_additive','non_language_share',
    'non_nested_R2_difference','within_direction_SD','between_direction_SD','total_row_SD'])]
display(gaps)
gaps.to_csv(RUN_DIR/'gaps_and_SD.csv',index=False)
components=ALL_RESULTS[ALL_RESULTS.component!='LM']
if len(components):display(components);components.to_csv(RUN_DIR/'EOS_and_punctuation_partitions.csv',index=False)
notes.extend([
    'All intervals are whole-meaning bootstrap percentile 95% intervals over 1,000 draws (or configured N_BOOT). Directions are fixed.',
    'The full-direction model and source+target+meaning model are not nested. Their difference is NOT the sentence-level variance fraction.',
    'Meaning share is a shared across-direction sentence effect; residual contains direction×meaning interaction plus noise.',
    'Within-direction and between-direction SDs use population denominators. Length results are exploratory point estimates.',
    'High R² among averaged direction means does not establish equally high sentence-level predictability.',
    'This held-dyad protocol retains the same meanings in other training directions; no unseen-meaning generalization claim.'
])
(RUN_DIR/'report_ready_summary.txt').write_text('\n\n'.join(notes),encoding='utf8')


# %% original code cell 5; notebook cell 11
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
left=np.zeros(2);models=list(INPUTS)
for metric,label,color in [('direction_share','Direction','#285B8A'),('meaning_share','Meaning','#B66822'),('residual_share','Residual / interaction + noise','#929292')]:
    values=np.array([100*entry(model,metric).estimate for model in models])
    ax.barh(models,values,left=left,label=label,color=color);left+=values
ax.set_xlim(0,100);ax.set_xlabel('Share of total sentence-level lexical-mass variance (%)')
ax.set_title('fp32 teacher-forced baseline: 450 meanings × 40 directions')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=1,frameon=False)
fig.savefig(RUN_DIR/'row_level_variance_partition.png',dpi=220,bbox_inches='tight')
plt.show()

import shutil
archive=Path(shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR))
print('Result archive:',archive)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print('Local Jupyter: open the result archive at the path above.')

