# Original executed notebook code, in original cell order.
# Requires Colab/IPython and original inputs; see docs/VALIDATION_EXPERIMENTS.md.

# %% original code cell 0; notebook cell 1
# 1. Mount Drive, CPU analysis paths, and L4-scoring configuration.
import os,sys,subprocess,json,time,platform
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
try:
    from IPython.display import display
except ImportError:
    display=print
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
BACKUP_ROOT=Path(os.environ.get('E2_BACKUP_ROOT','/content/drive/MyDrive/indic_40_backups'))
CONFIDENCE_ROOT=Path(os.environ.get('E2_CONFIDENCE_ROOT','/content/drive/MyDrive/indic_40_downstream_strong/confidence_v2'))
OUTPUT_ROOT=Path(os.environ.get('E2_OUTPUT_ROOT','/content/drive/MyDrive/indic_40_enhancements/E2_quality_comet_v1'))
METRIC_CACHE=Path(os.environ.get('E2_METRIC_CACHE','/content/drive/MyDrive/indic_40_metric_cache'))
STAGING=Path('/content/e2_quality_staging')
WORKER_ENV_ROOT=Path('/content/e2_comet_python312')
N_BOOT=10_000
BOOT_SEED=20261104
BOOT_BATCH_SIZE=32
COMET_BATCH_SIZE=16
COMET_CHUNK_ROWS=1024
assert N_BOOT==10_000,'Frozen final-report protocol; label any changed development run separately.'
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True);STAGING.mkdir(parents=True,exist_ok=True)
print('Original backup root:',BACKUP_ROOT)
print('Completed likelihood cache:',CONFIDENCE_ROOT)
print('New E2 output root:',OUTPUT_ROOT)
print('L4 stage: new COMET quality scores on saved outputs. CPU stage: both-model held-out regressions.')


# %% original code cell 1; notebook cell 2
"""E2 continuous quality prediction on saved generated-output attention. CPU analysis."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['pair','sentence_id','corpus_sentence_id']
METHODS=['A_likelihood','B_raw_LM','C_language_identity','D_signed_residual',
         'E_likelihood_raw_LM','F_likelihood_signed_residual',
         'G_controlled_likelihood','H_controlled_likelihood_raw_LM',
         'I_controlled_likelihood_signed_residual','J_controlled_likelihood_abs_residual',
         'K_controlled_likelihood_abs_raw_deviation']
METRICS=['Pearson_r','Spearman_rho','MAE','RMSE']
PROTOCOLS=['meaning_held_out','meaning_and_dyad_held_out']
CONFIDENCE_FEATURES=['hyp_mean_logprob','hyp_p10_logprob','hyp_min_logprob','hyp_std_logprob','hyp_sum_logprob','hyp_eos_logprob']

def progress(message):print(time.strftime('%H:%M:%S'),message,flush=True)
def require(condition,message):
    if not bool(condition):raise ValueError(message)
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(text):return hashlib.sha256(str(text).encode('utf-8')).hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def atomic_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,index=False);tmp.replace(path)
def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
def freeze_json(path,obj):
    path=Path(path)
    if path.exists():require(json.loads(path.read_text())==obj,f'Frozen settings differ: {path}; use a new version/output directory for changes.')
    else:atomic_json(path,obj)
def truth(s):return s.astype(str).str.strip().str.lower().isin(['true','1','1.0'])
def reverse(pair):return '__'.join(reversed(pair.split('__')))



# %% original code cell 2; notebook cell 4
def load_existing(study,backup_root,score_root,expected):
    dataset='holdout' if study=='it2' else 'nllb'
    dirname='indic_40_balanced_holdout_750_rows250_999_locked_v4' if study=='it2' else 'nllb_40_confirmatory_holdout_750_final_v6'
    root=Path(backup_root)/dirname
    paths={'attention':root/'free_generation_all_stats.csv',
           'quality':root/'manifests/sentence_quality_greedy.csv',
           'confidence':Path(score_root)/(dataset+'_logprob.csv'),
           'identity':root/'manifests/sample_identity.csv',
           'config':root/'manifests/run_config.json','source_lock':root/'manifests/source_lock.json'}
    for name,path in paths.items():require(path.is_file(),f'Missing {study} {name}: {path}. Use the completed original files; no model scoring is silently rerun.')
    hashes={}
    for name,path in paths.items():
        hashes[name]=sha_file(path)
        if name in expected[study]:require(hashes[name]==expected[study][name],f'{study}: {name} differs from the verified completed run.')
    config=json.loads(paths['config'].read_text())
    require(int(config['n_sentences'])==750,f'{study}: expected 750 historical meanings')
    require(('indictrans2' if study=='it2' else 'nllb') in config['model'].lower(),f'{study}: wrong original model')
    cols=set(KEY+['hyp_text','mean','src_len','tgt_len','analysis_view','generation_hit_max_length',
                   'generation_ended_with_eos','attention_inference_eligible','attention_tensor_role'])
    raw=pd.read_csv(paths['attention'],usecols=lambda c:c in cols,dtype={'corpus_sentence_id':str},keep_default_na=False)
    q=pd.read_csv(paths['quality'],dtype={'corpus_sentence_id':str})
    confidence=pd.read_csv(paths['confidence'],dtype={'corpus_sentence_id':str})
    identity=pd.read_csv(paths['identity'],dtype={'corpus_sentence_id':str})
    for frame,name in [(raw,'attention'),(q,'quality'),(confidence,'confidence'),(identity,'identity')]:
        require(set(KEY)<=set(frame),f'{study}: missing identity columns in {name}')
        require(len(frame)==30000 and not frame.duplicated(KEY).any(),f'{study}: wrong or duplicate {name} row count')
        require(frame.pair.nunique()==40 and frame.groupby('pair').corpus_sentence_id.nunique().eq(750).all(),f'{study}: incomplete {name} panel')
        require(set(frame.corpus_sentence_id)==set(map(str,range(250,1000))),f'{study}: {name} corpus identities differ')
    required_scores=CONFIDENCE_FEATURES+['hyp_scored_tokens','source_token_match','hyp_retokenization_match']
    require(set(required_scores)<=set(confidence),f'{study}: incomplete saved v2 likelihood columns')
    require(truth(confidence.source_token_match).all(),f'{study}: confidence source replay mismatch')
    require((confidence.hyp_scored_tokens>0).all(),f'{study}: empty scored hypotheses need an explicit new policy')
    score_cols=KEY+required_scores
    d=raw.merge(q[KEY+['sentence_chrfpp']],on=KEY,validate='one_to_one').merge(confidence[score_cols],on=KEY,validate='one_to_one')
    d=d.merge(identity[KEY+['source_sha256','target_sha256']],on=KEY,validate='one_to_one')
    require(len(d)==30000,f'{study}: row identities do not match across inputs')
    require(d.sentence_chrfpp.between(0,100).all(),f'{study}: invalid chrF++ values')
    for col in CONFIDENCE_FEATURES:
        if col=='hyp_eos_logprob':
            require(not np.isinf(d[col]).any(),f'{study}: infinite EOS likelihood');continue
        require(np.isfinite(d[col]).all(),f'{study}: nonfinite saved likelihood feature {col}')
    require((d.hyp_mean_logprob<=1e-4).all(),f'{study}: positive mean log likelihood')
    eligible=d.analysis_view.eq('lexical_no_punctuation')
    capped=truth(d.generation_hit_max_length)
    if study=='nllb':
        require(int(capped.sum())==57,'NLLB cap count differs from the verified 57 rows')
        require({'attention_inference_eligible','attention_tensor_role'}<=set(d),'NLLB eligibility columns missing')
        eligible &= ~capped & truth(d.attention_inference_eligible) & d.attention_tensor_role.eq('inferential')
    require(int(eligible.sum())=={'it2':29997,'nllb':29943}[study],f'{study}: eligible attention support changed')
    d['eligible_attention']=eligible;d['generation_capped']=capped
    d['LM']=np.where(eligible,pd.to_numeric(d['mean'],errors='coerce')*pd.to_numeric(d.src_len,errors='coerce'),np.nan)
    require(np.isfinite(d.loc[eligible,'LM']).all() and d.loc[eligible,'LM'].between(0,1+1e-5).all(),f'{study}: invalid eligible lexical mass')
    require((pd.to_numeric(d.loc[eligible,'src_len'])>0).all() and (pd.to_numeric(d.loc[eligible,'tgt_len'])>0).all(),f'{study}: invalid lexical lengths')
    d[['source','target']]=d.pair.str.split('__',expand=True)
    require(d.source.nunique()==d.target.nunique()==11,f'{study}: expected the original 11-language graph')
    d['study']=study
    d=d.sort_values(['pair','corpus_sentence_id'],key=lambda s:s.astype(int) if s.name=='corpus_sentence_id' else s).reset_index(drop=True)
    info={'study':study,'dataset':dataset,'rows':len(d),'eligible_attention_rows':int(eligible.sum()),
        'capped_rows':int(capped.sum()),'original_attention_collection_precision':'fp16',
        'confidence_protocol':'retokenized-saved-hypothesis-content-logprob-v2',
        'confidence_hypothesis_retokenization_match_fraction':float(truth(d.hyp_retokenization_match).mean()),
        'input_sha256':hashes,'input_paths':{k:str(v) for k,v in paths.items()}}
    return d,info

def make_meaning_split(ids,seed):
    ids=sorted(map(str,ids),key=int);require(len(ids)==750,'E2 expects 750 common meanings')
    shuffled=np.random.default_rng(seed).permutation(ids)
    split={'seed':seed,'train':sorted(map(str,shuffled[:525]),key=int),
        'validation':sorted(map(str,shuffled[525:637]),key=int),'test':sorted(map(str,shuffled[637:]),key=int)}
    require([len(split[k]) for k in ['train','validation','test']]==[525,112,113],'E2 split sizes changed')
    return split

def write_scoring_inputs(frames,staging):
    files={}
    for study,d in frames.items():
        out=d[KEY+['hyp_text','source','target','source_sha256','target_sha256']].copy()
        out['hyp_sha256']=out.hyp_text.map(digest)
        p=Path(staging)/(study+'_comet_inputs.csv');atomic_csv(p,out)
        files[study]={'path':str(p),'sha256':sha_file(p),'rows':len(out)}
    return files

def attach_comet(frame,score_path,completion_path,expected_metric):
    score_path=Path(score_path);completion=json.loads(Path(completion_path).read_text())
    require(completion['metric_repo']==expected_metric['repo'] and completion['metric_revision']==expected_metric['revision'],'COMET cache is from a different metric revision')
    require(completion['score_sha256']==sha_file(score_path),'COMET score file changed after completion')
    scores=pd.read_csv(score_path,dtype={'corpus_sentence_id':str})
    require(len(scores)==30000 and not scores.duplicated(KEY).any(),'Incomplete COMET output')
    columns=KEY+['comet','hyp_sha256','source_sha256','target_sha256','source_truncated','hypothesis_truncated','reference_truncated']
    require(set(columns)<=set(scores),'COMET score/audit columns missing')
    merged=frame.merge(scores[columns],on=KEY,validate='one_to_one',suffixes=('','_score'))
    require(len(merged)==len(frame),'COMET rows do not match the original hypotheses')
    require(merged.hyp_text.map(digest).eq(merged.hyp_sha256).all(),'COMET scored a different hypothesis text')
    for col in ['source_sha256','target_sha256']:
        require(merged[col].eq(merged[col+'_score']).all(),'COMET source/reference identity mismatch')
    require(np.isfinite(merged.comet).all(),'Nonfinite COMET scores; no silent dropping')
    return merged



# %% original code cell 3; notebook cell 6
def language_design(d,languages):
    n=len(languages);parts=[np.ones((len(d),1))]
    for role in ['source','target']:
        parts.append(np.column_stack([d[role].eq(l).to_numpy(float) for l in languages])-1/n)
    return np.column_stack(parts)

def calibration_fit(training,languages):
    require(len(training)>0,'Empty calibration training data')
    means=training.groupby(['pair','source','target'],sort=True,as_index=False).LM.mean()
    X=language_design(means,languages);rank=int(np.linalg.matrix_rank(X))
    require(rank==2*len(languages)-1,f'Language calibration is not identifiable: rank {rank}')
    coefficient=np.linalg.pinv(X)@means.LM.to_numpy(float)
    return coefficient,rank,means.pair.tolist()

def crossfit_residuals(training,evaluation,languages,n_folds,seed):
    """OOF residuals for quality-model training; full-training fit for evaluation."""
    ids=sorted(training.corpus_sentence_id.unique(),key=int)
    folds=np.array_split(np.random.default_rng(seed).permutation(ids),n_folds)
    residual=np.full(len(training),np.nan);fold_audit=[]
    for fi,held_ids in enumerate(folds):
        held=training.corpus_sentence_id.isin(set(held_ids)).to_numpy()
        coefficient,rank,pairs=calibration_fit(training.loc[~held],languages)
        residual[held]=training.loc[held,'LM'].to_numpy()-language_design(training.loc[held],languages)@coefficient
        fold_audit.append({'inner_fold':fi,'train_meanings':int(training.loc[~held].corpus_sentence_id.nunique()),
            'held_meanings':len(held_ids),'rank':rank,'held_id_sha256':digest('|'.join(sorted(map(str,held_ids),key=int)))})
    require(np.isfinite(residual).all(),'Cross-fit left unassigned residuals')
    coef,rank,pairs=calibration_fit(training,languages)
    evaluation_residual=evaluation.LM.to_numpy()-language_design(evaluation,languages)@coef
    fixed_training_residual=training.LM.to_numpy()-language_design(training,languages)@coef
    return residual,evaluation_residual,fixed_training_residual,coef,rank,pairs,fold_audit

def features(d,residual,languages,raw_center):
    L=language_design(d,languages)[:,1:]
    likelihood=d.hyp_mean_logprob.to_numpy(float)[:,None]
    raw=d.LM.to_numpy(float)[:,None];r=np.asarray(residual,float)[:,None]
    lengths=np.column_stack([np.log1p(d.src_len.to_numpy(float)),np.log1p(d.tgt_len.to_numpy(float)),
        np.abs(np.log((d.tgt_len.to_numpy(float)+1)/(d.src_len.to_numpy(float)+1)))])
    confidence=d[CONFIDENCE_FEATURES].to_numpy(float)
    controlled=np.column_stack([L,lengths,confidence,np.isnan(confidence).astype(float)])
    return {
        'A_likelihood':likelihood,'B_raw_LM':raw,'C_language_identity':L,'D_signed_residual':r,
        'E_likelihood_raw_LM':np.column_stack([likelihood,raw]),
        'F_likelihood_signed_residual':np.column_stack([likelihood,r]),
        'G_controlled_likelihood':controlled,
        'H_controlled_likelihood_raw_LM':np.column_stack([controlled,raw]),
        'I_controlled_likelihood_signed_residual':np.column_stack([controlled,r]),
        'J_controlled_likelihood_abs_residual':np.column_stack([controlled,np.abs(r)]),
        'K_controlled_likelihood_abs_raw_deviation':np.column_stack([controlled,np.abs(raw-raw_center)])}

def fit_linear(train_X,test_X,train_y):
    """Fixed OLS, training-only standardization; no tuning or test-statistic use."""
    require(not np.isinf(train_X).any() and not np.isinf(test_X).any() and np.isfinite(train_y).all(),'Infinite feature or nonfinite outcome')
    # Only saved confidence features may be missing. Attention rows were already
    # required complete. The feature builder supplies confidence missingness flags.
    prepared=np.array(train_X,float,copy=True);prepared[:,np.isnan(prepared).all(axis=0)]=0.
    median=np.nanmedian(prepared,axis=0)
    train_X=np.where(np.isnan(train_X),median,train_X);test_X=np.where(np.isnan(test_X),median,test_X)
    mu=train_X.mean(0);scale=train_X.std(0);scale=np.where(scale>1e-10,scale,1.)
    A=np.column_stack([np.ones(len(train_X)),(train_X-mu)/scale])
    B=np.column_stack([np.ones(len(test_X)),(test_X-mu)/scale])
    coefficient=np.linalg.lstsq(A,train_y,rcond=None)[0]
    return B@coefficient,{'rank':int(np.linalg.matrix_rank(A)),'columns':A.shape[1],
        'coefficient':coefficient.tolist(),'training_feature_mean':mu.tolist(),'training_feature_scale':scale.tolist(),
        'training_imputation_median':median.tolist(),'imputation':'confidence only; training median, all-missing training column zero; explicit missingness indicators'}

def fit_protocol(frame,split,protocol,output,seed):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    eligible=frame[frame.eligible_attention].copy()
    training=eligible[eligible.corpus_sentence_id.isin(split['train'])].reset_index(drop=True)
    evaluation=eligible[eligible.corpus_sentence_id.isin(split['validation']+split['test'])].reset_index(drop=True)
    require(set(training.corpus_sentence_id).isdisjoint(evaluation.corpus_sentence_id),'Meaning leakage')
    pairs=sorted(frame.pair.unique());languages=sorted(frame.source.unique())
    require(len(pairs)==40 and all(reverse(p) in pairs for p in pairs),'Expected the original reciprocal graph')
    dyads=[p for p in pairs if p<reverse(p)]
    folds=[('all_directions',set(pairs))] if protocol=='meaning_held_out' else [(p,{p,reverse(p)}) for p in dyads]
    rows=[];audits=[];equivalence=[];models=[]
    for fold_index,(held_name,eval_pairs) in enumerate(folds):
        train=training if protocol=='meaning_held_out' else training[~training.pair.isin(eval_pairs)]
        test=evaluation[evaluation.pair.isin(eval_pairs)]
        train=train.reset_index(drop=True);test=test.reset_index(drop=True)
        require(len(train)>0 and len(test)>0,'Empty evaluation fold')
        if protocol!='meaning_held_out':require(not set(train.pair)&eval_pairs,'Held-dyad leakage')
        rtrain,rtest,rfixed,coef,rank,cal_pairs,inner=crossfit_residuals(train,test,languages,5,seed)
        if protocol!='meaning_held_out':require(not set(cal_pairs)&eval_pairs,'Calibration used held-dyad rows')
        center=float(train.LM.mean())
        Xtrain=features(train,rtrain,languages,center);Xtest=features(test,rtest,languages,center)
        ytrain=train[['comet','sentence_chrfpp']].to_numpy(float)
        prediction=test[KEY+['source','target','LM','hyp_mean_logprob','comet','sentence_chrfpp',
            'generation_capped','source_truncated','hypothesis_truncated','reference_truncated']].copy()
        prediction['residual']=rtest;prediction['abs_residual']=np.abs(rtest)
        prediction['partition']=np.where(prediction.corpus_sentence_id.isin(split['test']),'test','validation')
        prediction['protocol']=protocol;prediction['fold']=fold_index;prediction['held_dyad']=held_name
        for method in METHODS:
            values,model=fit_linear(Xtrain[method],Xtest[method],ytrain)
            prediction['comet__'+method]=values[:,0];prediction['chrfpp__'+method]=values[:,1]
            models.append({'fold':fold_index,'method':method,**model})
        # Same fitted additive baseline + language indicators is a linear
        # reparameterization. This diagnostic uses in-sample training residuals
        # only to verify algebra; all reported residual methods above use OOF.
        fixed=features(train,rfixed,languages,center)
        a,_=fit_linear(fixed['H_controlled_likelihood_raw_LM'],Xtest['H_controlled_likelihood_raw_LM'],ytrain)
        b,_=fit_linear(fixed['I_controlled_likelihood_signed_residual'],Xtest['I_controlled_likelihood_signed_residual'],ytrain)
        error=float(np.max(np.abs(a-b)))
        require(error<1e-6,f'Linear reparameterization audit failed: {error}')
        equivalence.append({'fold':fold_index,'held_dyad':held_name,'max_prediction_difference':error,
            'scope':'same-fit signed residual with language controls; not an added-information claim'})
        for role,names in [('validation',split['validation']),('test',split['test'])]:
            count=test[test.corpus_sentence_id.isin(names)]
            audits.append({'fold':fold_index,'held_dyad':held_name,'partition':role,
                'training_rows':len(train),'training_meanings':train.corpus_sentence_id.nunique(),
                'evaluation_rows':len(count),'evaluation_meanings':count.corpus_sentence_id.nunique(),
                'calibration_rank':rank,'training_pair_count':train.pair.nunique(),
                'training_meaning_sha256':digest('|'.join(sorted(train.corpus_sentence_id.unique(),key=int))),
                'inner_folds':json.dumps(inner)})
        rows.append(prediction)
        progress(f'{frame.study.iloc[0]} {protocol}: fold {fold_index+1}/{len(folds)} complete')
    predictions=pd.concat(rows,ignore_index=True)
    require(not predictions.duplicated(KEY).any(),'Duplicate out-of-fold predictions')
    require(len(predictions)==len(evaluation),'Evaluation coverage mismatch')
    atomic_csv(output/'predictions.csv',predictions)
    atomic_csv(output/'fold_audit.csv',pd.DataFrame(audits));atomic_csv(output/'linear_reparameterization_audit.csv',pd.DataFrame(equivalence))
    atomic_json(output/'fitted_quality_models.json',models)
    return predictions



# %% original code cell 4; notebook cell 8
def weighted_correlation(x,y,w):
    w=np.asarray(w,float);n=w.sum(1);mx=np.sum(w*x,axis=1)/n;my=np.sum(w*y,axis=1)/n
    dx=x-mx[:,None];dy=y-my[:,None]
    vx=np.sum(w*dx*dx,axis=1);vy=np.sum(w*dy*dy,axis=1)
    den=np.sqrt(vx*vy)
    # Floating summation can leave tiny nonzero variance for constant predictions.
    tolerance=16*np.finfo(float).eps**2
    valid=(vx>tolerance*np.maximum(np.sum(w*x*x,axis=1),1.)) & (vy>tolerance*np.maximum(np.sum(w*y*y,axis=1),1.))
    return np.divide(np.sum(w*dx*dy,axis=1),den,out=np.full(len(n),np.nan),where=valid)

def rank_structure(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');ordered=values[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    groups=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(values)]))
    inverse=np.empty(len(order),int);inverse[order]=np.arange(len(order))
    return order,starts,groups[inverse]

def weighted_midranks(weights,structure):
    order,starts,row_group=structure
    counts=np.add.reduceat(weights[:,order],starts,axis=1)
    mid=np.cumsum(counts,axis=1)-counts/2
    return mid[:,row_group]

def metric_batch(observed,predicted,weights,y_structure=None,p_structure=None):
    y=np.asarray(observed,float);p=np.asarray(predicted,float);w=np.asarray(weights,float)
    total=w.sum(1)
    y_rank=weighted_midranks(w,y_structure or rank_structure(y))
    p_rank=weighted_midranks(w,p_structure or rank_structure(p))
    diff=p-y
    return np.column_stack([weighted_correlation(p[None,:],y[None,:],w),
        weighted_correlation(p_rank,y_rank,w),w@np.abs(diff)/total,np.sqrt(w@(diff*diff)/total)])

def point_tables(predictions):
    rows=[];dyads=[]
    for partition,g in predictions.groupby('partition',sort=False):
        for outcome,column in [('comet','comet'),('chrfpp','sentence_chrfpp')]:
            for method in METHODS:
                p=g[outcome+'__'+method].to_numpy();y=g[column].to_numpy()
                values=metric_batch(y,p,np.ones((1,len(g))))[0]
                rows.append({'partition':partition,'outcome':outcome,'method':method,'rows':len(g),
                    'meanings':g.corpus_sentence_id.nunique(),**dict(zip(METRICS,values))})
                for pair,h in g.groupby('pair'):
                    v=metric_batch(h[column].to_numpy(),h[outcome+'__'+method].to_numpy(),np.ones((1,len(h))))[0]
                    dyads.append({'partition':partition,'outcome':outcome,'method':method,'pair':pair,'rows':len(h),**dict(zip(METRICS,v))})
    return pd.DataFrame(rows),pd.DataFrame(dyads)

def bootstrap_predictions(test,split,n_boot,seed,batch_size,path,fingerprint):
    ids=list(split['test']);position={x:i for i,x in enumerate(ids)}
    row_ids=np.array([position[x] for x in test.corpus_sentence_id],int)
    keys=[(outcome,method) for outcome in ['comet','chrfpp'] for method in METHODS]
    samples=np.full((n_boot,len(keys),len(METRICS)),np.nan);rng=np.random.default_rng(seed);completed=0;chain=''
    if Path(path).exists():
        with np.load(path,allow_pickle=False) as z:
            require(str(z['fingerprint'].item())==fingerprint,'Bootstrap cache input/settings differ')
            require(z['samples'].shape==samples.shape,'Bootstrap dimensions changed')
            samples[:]=z['samples'];completed=int(z['completed'].item());chain=str(z['draw_chain'].item())
            rng.bit_generator.state=json.loads(str(z['rng_state'].item()))
        progress(f'Resuming evaluation bootstrap {completed:,}/{n_boot:,}')
    prepared={}
    for outcome,method in keys:
        y=test['comet' if outcome=='comet' else 'sentence_chrfpp'].to_numpy(float)
        p=test[outcome+'__'+method].to_numpy(float)
        prepared[(outcome,method)]=(y,p,rank_structure(y),rank_structure(p))
    started=time.perf_counter()
    for start in range(completed,n_boot,batch_size):
        stop=min(start+batch_size,n_boot);b=stop-start
        meaning_weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=b)
        weights=meaning_weights[:,row_ids]
        chain=digest(chain+hashlib.sha256(meaning_weights.tobytes()).hexdigest())
        for ki,key in enumerate(keys):
            y,p,ys,ps=prepared[key]
            samples[start:stop,ki]=metric_batch(y,p,weights,ys,ps)
        if start==completed or stop==n_boot or stop//1000>start//1000:
            atomic_npz(path,samples=samples,completed=np.array(stop),fingerprint=np.array(fingerprint),
                rng_state=np.array(json.dumps(rng.bit_generator.state)),draw_chain=np.array(chain),
                keys=np.array(['|'.join(k) for k in keys]),metrics=np.array(METRICS))
            progress(f'Evaluation bootstrap {stop:,}/{n_boot:,}; {time.perf_counter()-started:.1f}s this session')
    return samples,keys,chain

def intervals_and_contrasts(point,samples,keys):
    output=point.copy();contrasts=[]
    for i,row in output.iterrows():
        if row.partition!='test':continue
        ki=keys.index((row.outcome,row.method))
        for mi,metric in enumerate(METRICS):
            v=samples[:,ki,mi];v=v[np.isfinite(v)]
            lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
            output.loc[i,metric+'_ci_low']=lo;output.loc[i,metric+'_ci_high']=hi
            output.loc[i,metric+'_bootstrap_valid']=len(v)
    comparisons=[('F_likelihood_signed_residual','A_likelihood','document comparison'),
        ('E_likelihood_raw_LM','A_likelihood','document raw-attention comparison'),
        ('I_controlled_likelihood_signed_residual','G_controlled_likelihood','primary controlled added-value comparison'),
        ('H_controlled_likelihood_raw_LM','G_controlled_likelihood','matched raw-attention comparison'),
        ('J_controlled_likelihood_abs_residual','G_controlled_likelihood','prespecified secondary absolute-residual comparison'),
        ('J_controlled_likelihood_abs_residual','K_controlled_likelihood_abs_raw_deviation','matched nonlinear-feature comparison')]
    for outcome in ['comet','chrfpp']:
        for a,b,role in comparisons:
            ra=point[(point.partition=='test')&(point.outcome==outcome)&(point.method==a)].iloc[0]
            rb=point[(point.partition=='test')&(point.outcome==outcome)&(point.method==b)].iloc[0]
            for mi,metric in enumerate(METRICS):
                sign=1 if mi<2 else -1
                v=sign*(samples[:,keys.index((outcome,a)),mi]-samples[:,keys.index((outcome,b)),mi]);v=v[np.isfinite(v)]
                lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
                contrasts.append({'outcome':outcome,'method':a,'comparator':b,'role':role,'metric':metric,
                    'estimate':sign*(ra[metric]-rb[metric]),'ci_low':lo,'ci_high':hi,'bootstrap_valid':len(v),
                    'positive_means':'higher correlation' if mi<2 else 'lower error'})
    return output,pd.DataFrame(contrasts)



# %% original code cell 5; notebook cell 10
def run_quality_study(frame,split,output,n_boot,seed,batch_size,code_hash,metric_signature):
    study=frame.study.iloc[0];all_summary=[];all_contrasts=[]
    for protocol in PROTOCOLS:
        folder=Path(output)/study/protocol;folder.mkdir(parents=True,exist_ok=True)
        signature={'study':study,'protocol':protocol,'code_sha256':code_hash,'split':split,
            'n_boot':n_boot,'seed':seed,'batch_size':batch_size,'metric_signature':metric_signature,
            'row_input_digest':digest(frame[KEY+['LM','comet','sentence_chrfpp','hyp_mean_logprob','eligible_attention']].to_csv(index=False))}
        fingerprint=digest(json.dumps(signature,sort_keys=True));freeze_json(folder/'analysis_lock.json',signature)
        prediction_path=folder/'predictions.csv';fit_stamp=folder/'fit_complete.json'
        if fit_stamp.exists():
            old=json.loads(fit_stamp.read_text());require(old['fingerprint']==fingerprint,'Fitted prediction cache differs')
            require(old['predictions_sha256']==sha_file(prediction_path),'Fitted predictions were modified')
            predictions=pd.read_csv(prediction_path,dtype={'corpus_sentence_id':str})
            progress(f'{study} {protocol}: reusing verified fitted predictions')
        else:
            predictions=fit_protocol(frame,split,protocol,folder,split['seed'])
            atomic_json(fit_stamp,{'fingerprint':fingerprint,'predictions_sha256':sha_file(prediction_path)})
        point,by_direction=point_tables(predictions)
        atomic_csv(folder/'point_metrics.csv',point);atomic_csv(folder/'per_direction_metrics.csv',by_direction)
        test=predictions[predictions.partition=='test'].reset_index(drop=True)
        samples,keys,chain=bootstrap_predictions(test,split,n_boot,seed,batch_size,folder/'bootstrap_checkpoint.npz',fingerprint)
        summary,contrasts=intervals_and_contrasts(point,samples,keys)
        summary['study']=study;summary['protocol']=protocol;contrasts['study']=study;contrasts['protocol']=protocol
        atomic_csv(folder/'metrics_with_CI.csv',summary);atomic_csv(folder/'paired_contrasts.csv',contrasts)
        atomic_json(folder/'completed.json',{'fingerprint':fingerprint,'bootstrap_completed':n_boot,'draw_chain':chain,
            'ci_scope':'test-meaning sampling conditional on fixed fitted models; training-fit uncertainty omitted',
            'result_sha256':{name:sha_file(folder/name) for name in ['predictions.csv','metrics_with_CI.csv','paired_contrasts.csv','bootstrap_checkpoint.npz']}})
        all_summary.append(summary);all_contrasts.append(contrasts)
    return pd.concat(all_summary,ignore_index=True),pd.concat(all_contrasts,ignore_index=True)


# %% original code cell 6; notebook cell 11
# 2. Frozen design, data checksums and metric identity.
EXPECTED_INPUTS=json.loads('{"it2": {"attention": "3cd06bef9d07d524583b7aadf108c034e86bd509273f3c47c4ba43648c8e4ef4", "quality": "f1e490ce9e3ca504b595530cbf598576a758601dbaddc822af8852903075467a", "confidence": "4d84e5c5504a1c05ea4a65fcaf50415042b4c3732cb8a5f48feb470553b4a0ee", "identity": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "source_lock": "8e616f449351528c36e04f271c29565c366baa79022d8b7a809c2de5fccca62d"}, "nllb": {"attention": "28362cdce9b8981b83fd25d3e793fa376c5b5aaaec99a8940474650b50b8b93a", "quality": "59237d9dca4f064f19c549b42a696a885f314739c799fc5acd3f47e2164330f3", "confidence": "7255d23bcf899d14440ae76d9f29dd12f2d640ffd336429c21ad7b3f5b537cd0", "identity": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "source_lock": "2126ac90c3558019c31a671f485395b1b6497d6df068751c5f761cf8252da644"}}')
FROZEN_SPLIT=json.loads('{"seed": 20261103, "train": ["250", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "263", "264", "265", "266", "268", "270", "271", "272", "274", "275", "276", "277", "278", "279", "284", "285", "286", "287", "288", "289", "290", "292", "293", "294", "296", "297", "298", "299", "300", "301", "302", "304", "305", "306", "308", "309", "310", "312", "314", "315", "318", "320", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "333", "335", "336", "337", "338", "339", "340", "342", "343", "345", "347", "348", "349", "350", "351", "352", "353", "354", "356", "357", "359", "360", "361", "362", "364", "365", "366", "368", "369", "370", "371", "372", "373", "375", "376", "378", "379", "380", "381", "382", "384", "385", "386", "387", "388", "389", "391", "392", "393", "394", "395", "399", "401", "405", "407", "408", "409", "410", "411", "412", "414", "415", "417", "419", "420", "421", "422", "424", "426", "427", "428", "430", "432", "433", "434", "437", "439", "440", "441", "442", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "455", "456", "457", "458", "461", "462", "463", "464", "465", "466", "467", "468", "469", "471", "473", "474", "475", "478", "479", "480", "481", "483", "485", "486", "489", "490", "491", "492", "494", "495", "496", "500", "501", "502", "505", "508", "510", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "526", "529", "530", "532", "536", "538", "539", "540", "542", "544", "545", "546", "547", "549", "550", "554", "555", "556", "558", "559", "560", "561", "564", "567", "568", "571", "572", "573", "574", "575", "576", "577", "578", "579", "580", "581", "584", "585", "586", "587", "588", "592", "593", "594", "595", "596", "597", "598", "599", "600", "602", "603", "604", "605", "607", "608", "609", "610", "611", "612", "613", "615", "616", "617", "619", "623", "624", "625", "628", "629", "630", "631", "632", "634", "635", "636", "637", "638", "639", "641", "644", "645", "647", "648", "650", "652", "653", "654", "655", "656", "657", "658", "659", "661", "662", "663", "665", "668", "669", "671", "672", "675", "676", "677", "678", "680", "681", "683", "684", "685", "686", "688", "689", "691", "693", "695", "696", "697", "700", "701", "702", "703", "705", "706", "710", "711", "712", "713", "714", "715", "716", "717", "718", "720", "721", "722", "724", "725", "726", "727", "728", "729", "730", "734", "735", "736", "738", "739", "741", "742", "744", "745", "747", "749", "753", "754", "755", "756", "757", "758", "760", "761", "762", "764", "765", "766", "767", "768", "769", "770", "771", "773", "774", "775", "777", "778", "779", "782", "784", "786", "787", "790", "791", "792", "793", "795", "797", "798", "799", "800", "801", "802", "803", "804", "805", "810", "811", "812", "813", "816", "817", "818", "821", "822", "823", "824", "825", "826", "828", "835", "837", "839", "841", "843", "845", "846", "847", "848", "849", "851", "852", "853", "854", "855", "856", "857", "859", "860", "861", "863", "864", "865", "866", "868", "869", "870", "871", "873", "874", "875", "877", "878", "879", "880", "883", "884", "885", "886", "888", "889", "894", "895", "896", "897", "898", "899", "900", "901", "903", "904", "905", "906", "907", "911", "912", "913", "914", "916", "917", "918", "919", "920", "921", "924", "925", "926", "928", "931", "933", "934", "938", "939", "940", "941", "942", "945", "946", "947", "948", "949", "951", "952", "953", "955", "956", "958", "959", "963", "964", "965", "966", "967", "968", "970", "971", "972", "973", "974", "975", "977", "978", "980", "982", "984", "985", "986", "989", "991", "992", "994", "996", "997"], "validation": ["251", "262", "273", "281", "283", "295", "316", "319", "321", "332", "334", "341", "344", "346", "355", "363", "374", "377", "406", "413", "423", "425", "429", "438", "460", "470", "482", "484", "487", "488", "499", "503", "507", "511", "528", "534", "535", "537", "541", "548", "551", "552", "553", "562", "566", "569", "570", "583", "591", "614", "618", "620", "627", "642", "643", "666", "674", "690", "692", "694", "698", "699", "709", "731", "732", "733", "737", "743", "748", "751", "752", "772", "776", "781", "783", "785", "788", "794", "796", "809", "814", "819", "827", "829", "831", "832", "834", "840", "842", "850", "858", "872", "876", "881", "882", "887", "890", "902", "909", "910", "923", "929", "932", "935", "943", "944", "950", "962", "979", "981", "993", "995"], "test": ["267", "269", "280", "282", "291", "303", "307", "311", "313", "317", "358", "367", "383", "390", "396", "397", "398", "400", "402", "403", "404", "416", "418", "431", "435", "436", "443", "454", "459", "472", "476", "477", "493", "497", "498", "504", "506", "509", "525", "527", "531", "533", "543", "557", "563", "565", "582", "589", "590", "601", "606", "621", "622", "626", "633", "640", "646", "649", "651", "660", "664", "667", "670", "673", "679", "682", "687", "704", "707", "708", "719", "723", "740", "746", "750", "759", "763", "780", "789", "806", "807", "808", "815", "820", "830", "833", "836", "838", "844", "862", "867", "891", "892", "893", "908", "915", "922", "927", "930", "936", "937", "954", "957", "960", "961", "969", "976", "983", "987", "988", "990", "998", "999"]}')
FROZEN_PROTOCOL=json.loads('{"experiment": "E2", "version": "1.1", "prepared_on": "2026-11-01", "design_status": "retrospective analysis of previously generated translations; new COMET labels; no untouched-corpus claim", "studies": ["it2", "nllb"], "meanings": 750, "split": {"seed": 20261103, "train": ["250", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "263", "264", "265", "266", "268", "270", "271", "272", "274", "275", "276", "277", "278", "279", "284", "285", "286", "287", "288", "289", "290", "292", "293", "294", "296", "297", "298", "299", "300", "301", "302", "304", "305", "306", "308", "309", "310", "312", "314", "315", "318", "320", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "333", "335", "336", "337", "338", "339", "340", "342", "343", "345", "347", "348", "349", "350", "351", "352", "353", "354", "356", "357", "359", "360", "361", "362", "364", "365", "366", "368", "369", "370", "371", "372", "373", "375", "376", "378", "379", "380", "381", "382", "384", "385", "386", "387", "388", "389", "391", "392", "393", "394", "395", "399", "401", "405", "407", "408", "409", "410", "411", "412", "414", "415", "417", "419", "420", "421", "422", "424", "426", "427", "428", "430", "432", "433", "434", "437", "439", "440", "441", "442", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "455", "456", "457", "458", "461", "462", "463", "464", "465", "466", "467", "468", "469", "471", "473", "474", "475", "478", "479", "480", "481", "483", "485", "486", "489", "490", "491", "492", "494", "495", "496", "500", "501", "502", "505", "508", "510", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "526", "529", "530", "532", "536", "538", "539", "540", "542", "544", "545", "546", "547", "549", "550", "554", "555", "556", "558", "559", "560", "561", "564", "567", "568", "571", "572", "573", "574", "575", "576", "577", "578", "579", "580", "581", "584", "585", "586", "587", "588", "592", "593", "594", "595", "596", "597", "598", "599", "600", "602", "603", "604", "605", "607", "608", "609", "610", "611", "612", "613", "615", "616", "617", "619", "623", "624", "625", "628", "629", "630", "631", "632", "634", "635", "636", "637", "638", "639", "641", "644", "645", "647", "648", "650", "652", "653", "654", "655", "656", "657", "658", "659", "661", "662", "663", "665", "668", "669", "671", "672", "675", "676", "677", "678", "680", "681", "683", "684", "685", "686", "688", "689", "691", "693", "695", "696", "697", "700", "701", "702", "703", "705", "706", "710", "711", "712", "713", "714", "715", "716", "717", "718", "720", "721", "722", "724", "725", "726", "727", "728", "729", "730", "734", "735", "736", "738", "739", "741", "742", "744", "745", "747", "749", "753", "754", "755", "756", "757", "758", "760", "761", "762", "764", "765", "766", "767", "768", "769", "770", "771", "773", "774", "775", "777", "778", "779", "782", "784", "786", "787", "790", "791", "792", "793", "795", "797", "798", "799", "800", "801", "802", "803", "804", "805", "810", "811", "812", "813", "816", "817", "818", "821", "822", "823", "824", "825", "826", "828", "835", "837", "839", "841", "843", "845", "846", "847", "848", "849", "851", "852", "853", "854", "855", "856", "857", "859", "860", "861", "863", "864", "865", "866", "868", "869", "870", "871", "873", "874", "875", "877", "878", "879", "880", "883", "884", "885", "886", "888", "889", "894", "895", "896", "897", "898", "899", "900", "901", "903", "904", "905", "906", "907", "911", "912", "913", "914", "916", "917", "918", "919", "920", "921", "924", "925", "926", "928", "931", "933", "934", "938", "939", "940", "941", "942", "945", "946", "947", "948", "949", "951", "952", "953", "955", "956", "958", "959", "963", "964", "965", "966", "967", "968", "970", "971", "972", "973", "974", "975", "977", "978", "980", "982", "984", "985", "986", "989", "991", "992", "994", "996", "997"], "validation": ["251", "262", "273", "281", "283", "295", "316", "319", "321", "332", "334", "341", "344", "346", "355", "363", "374", "377", "406", "413", "423", "425", "429", "438", "460", "470", "482", "484", "487", "488", "499", "503", "507", "511", "528", "534", "535", "537", "541", "548", "551", "552", "553", "562", "566", "569", "570", "583", "591", "614", "618", "620", "627", "642", "643", "666", "674", "690", "692", "694", "698", "699", "709", "731", "732", "733", "737", "743", "748", "751", "752", "772", "776", "781", "783", "785", "788", "794", "796", "809", "814", "819", "827", "829", "831", "832", "834", "840", "842", "850", "858", "872", "876", "881", "882", "887", "890", "902", "909", "910", "923", "929", "932", "935", "943", "944", "950", "962", "979", "981", "993", "995"], "test": ["267", "269", "280", "282", "291", "303", "307", "311", "313", "317", "358", "367", "383", "390", "396", "397", "398", "400", "402", "403", "404", "416", "418", "431", "435", "436", "443", "454", "459", "472", "476", "477", "493", "497", "498", "504", "506", "509", "525", "527", "531", "533", "543", "557", "563", "565", "582", "589", "590", "601", "606", "621", "622", "626", "633", "640", "646", "649", "651", "660", "664", "667", "670", "673", "679", "682", "687", "704", "707", "708", "719", "723", "740", "746", "750", "759", "763", "780", "789", "806", "807", "808", "815", "820", "830", "833", "836", "838", "844", "862", "867", "891", "892", "893", "908", "915", "922", "927", "930", "936", "937", "954", "957", "960", "961", "969", "976", "983", "987", "988", "990", "998", "999"]}, "attention": "saved historical fp16 free-generation lexical mass; no reference-TF attention used as generated-output attention", "likelihood": "verified v2 content likelihood of retokenized saved hypotheses; not exact original sampling-path token likelihood", "metric": {"repo": "Unbabel/wmt22-comet-da", "revision": "2760a223ac957f30acfb18c8aa649b01cf1d75f2", "checkpoint_sha256": "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af", "encoder_repo": "FacebookAI/xlm-roberta-large", "encoder_revision": "c23d21b0620b635a76227c604d44e43a9f0ee389"}, "primary_outcome": "COMET22-DA", "secondary_outcome": "saved sentence chrF++", "score_all_generated_rows": 60000, "attention_eligible_rows": {"it2": 29997, "nllb": 29943}, "exclusions": "all original quality rows scored; fixed original attention eligibility applied equally to every predictive method; capped/unavailable attention not imputed", "models": "fixed OLS with training-only feature standardization; no hyperparameter search, clipping, or validation tuning", "calibration": "training-directed-mean source+target additive LM fit; 5-fold meaning-crossfitted training residuals; full-training baseline for validation/test", "protocols": ["meaning_held_out", "meaning_and_dyad_held_out"], "dyad_protocol": "20 folds; both reciprocal directions excluded from fitting and all inner calibration folds", "feature_sets": "A-F document sets; G-I matched language/length/likelihood controls; J absolute-residual and K matched raw-deviation secondary features", "rich_control_baseline": "all six saved likelihood summaries, source/target identity, source/output length controls; training-only confidence imputation with missingness flags", "primary_contrast": "I_controlled_likelihood_signed_residual vs G_controlled_likelihood on strict held-out rows, COMET Spearman rho", "secondary_prespecified": "absolute-residual J vs baseline G and matched nonlinear raw-deviation K; all four metrics and both outcomes reported", "metrics": ["Pearson_r", "Spearman_rho", "MAE", "RMSE"], "bootstrap": {"repetitions": 10000, "seed": 20261104, "batch_size": 32, "unit": "test meanings jointly across all available direction rows", "scope": "conditional on fitted quality models and training calibration; training-fit uncertainty not included", "Spearman": "weighted midranks recomputed for each resampled empirical distribution; ties/duplicate rows handled exactly", "pairing": "identical test-meaning draws for all methods, outcomes, protocols and systems"}, "linear_identity_warning": "with language indicators, same-fit signed LM residual is a reparameterization; raw/residual representations do not create new input information", "metric_truncation": "retain official metric truncation, record token lengths/flags and report coverage; no silent exclusion or replacement", "human_review": "generate a blinded 220-row validation-only template; actual human ratings remain optional and unperformed", "no_hallucination_label": true, "no_github_changes": true}')
METRIC_SPEC=json.loads('{"repo": "Unbabel/wmt22-comet-da", "revision": "2760a223ac957f30acfb18c8aa649b01cf1d75f2", "checkpoint_sha256": "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af", "encoder_repo": "FacebookAI/xlm-roberta-large", "encoder_revision": "c23d21b0620b635a76227c604d44e43a9f0ee389"}')
CORE_SHA256='ae2ad43b4240958dfbbd2fe73df1fff88fa6e6e5191963be40ffaca20ae5c4a8'
WORKER_SHA256='d33d685d6df505349d075cf8f31abc00c332dbd7397d0b4142e5bbe9043f8d2f'
WORKER_SOURCE='"""Isolated Python 3.12/L4 worker: score saved hypotheses with pinned COMET."""\nimport argparse\nimport gc\nimport hashlib\nimport importlib.metadata\nimport json\nimport os\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n\nKEY=[\'pair\',\'sentence_id\',\'corpus_sentence_id\']\nLANGS=[\'asm_Beng\',\'ben_Beng\',\'guj_Gujr\',\'hin_Deva\',\'kan_Knda\',\'mal_Mlym\',\'mar_Deva\',\'ory_Orya\',\'pan_Guru\',\'tam_Taml\',\'tel_Telu\']\n\ndef progress(message):print(time.strftime(\'%H:%M:%S\'),message,flush=True)\ndef require(condition,message):\n    if not bool(condition):raise ValueError(message)\ndef sha_file(path):\n    h=hashlib.sha256()\n    with Path(path).open(\'rb\') as f:\n        for chunk in iter(lambda:f.read(1024*1024),b\'\'):h.update(chunk)\n    return h.hexdigest()\ndef digest(text):return hashlib.sha256(str(text).encode(\'utf-8\')).hexdigest()\ndef atomic_json(path,obj):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(\'.tmp\')\n    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+\'\\n\',encoding=\'utf-8\');tmp.replace(path)\ndef atomic_csv(path,frame):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(\'.tmp\')\n    frame.to_csv(tmp,index=False);tmp.replace(path)\ndef freeze(path,obj):\n    if Path(path).exists():require(json.loads(Path(path).read_text())==obj,f\'Existing metric cache signature differs: {path}\')\n    else:atomic_json(path,obj)\n\ndef load_original_texts(config):\n    """Use an existing cache or recover only the original pinned FLORES corpus."""\n    cache=Path(config[\'text_cache\']);meta=cache.with_suffix(\'.manifest.json\')\n    candidates=[cache]+[Path(p) for p in config[\'existing_text_caches\']]\n    for path in candidates:\n        if not path.is_file():continue\n        wide=pd.read_csv(path,dtype={\'id\':str},keep_default_na=False)\n        if \'id\' in wide and set(LANGS)<=set(wide):\n            require(not wide.id.duplicated().any(),\'Duplicate original corpus IDs in text cache\')\n            progress(f\'Using saved original text cache: {path}\')\n            return wide.set_index(\'id\')\n    progress(\'Recovering the same pinned FLORES+ source/reference texts; no new corpus is introduced.\')\n    from datasets import load_dataset\n    from huggingface_hub import get_token\n    token=os.environ.get(\'HF_TOKEN\') or get_token()\n    require(token,\'The original FLORES+ corpus is gated. Enable the approved HF_TOKEN Colab secret; the token is never saved or printed.\')\n    dataset=load_dataset(config[\'dataset\'],revision=config[\'dataset_revision\'],split=\'devtest\',token=token)\n    long=dataset.to_pandas();long[\'lang_code\']=long.iso_639_3.astype(str)+\'_\'+long.iso_15924.astype(str)\n    sub=long[long.lang_code.isin(LANGS)]\n    require(not sub.duplicated([\'id\',\'lang_code\']).any(),\'Duplicate corpus records\')\n    wide=sub.pivot(index=\'id\',columns=\'lang_code\',values=\'text\').reset_index()\n    wide[\'id\']=wide.id.astype(str)\n    require(set(LANGS)<=set(wide) and not wide[LANGS].isna().any().any(),\'Incomplete original-language text cache\')\n    atomic_csv(cache,wide)\n    atomic_json(meta,{\'dataset\':config[\'dataset\'],\'revision\':config[\'dataset_revision\'],\'sha256\':sha_file(cache),\n                     \'scope\':\'Original authorized corpus texts; private cache, excluded from result archive.\'})\n    return wide.set_index(\'id\')\n\ndef prepare_records(frame,wide):\n    sources=[];references=[]\n    for row in frame.itertuples(index=False):\n        require(str(row.corpus_sentence_id) in wide.index,\'Saved meaning is missing from the original text cache\')\n        source=str(wide.loc[str(row.corpus_sentence_id),row.source])\n        reference=str(wide.loc[str(row.corpus_sentence_id),row.target])\n        require(digest(source)==row.source_sha256,\'Original source text hash mismatch; corpus substitution is forbidden\')\n        require(digest(reference)==row.target_sha256,\'Original reference text hash mismatch; corpus substitution is forbidden\')\n        require(digest(row.hyp_text)==row.hyp_sha256,\'Saved generated text hash mismatch\')\n        sources.append(source);references.append(reference)\n    return [{\'src\':s,\'mt\':h,\'ref\':r} for s,h,r in zip(sources,frame.hyp_text.astype(str),references)]\n\ndef score_adaptive(model,records,batch_size):\n    import torch\n    current=batch_size\n    while True:\n        try:\n            result=model.predict(records,batch_size=current,gpus=1,accelerator=\'gpu\',num_workers=0,\n                progress_bar=False,length_batching=True,mc_dropout=0)\n            scores=np.asarray(result.scores,dtype=float)\n            require(len(scores)==len(records) and np.isfinite(scores).all(),\'COMET returned incomplete/nonfinite segment scores\')\n            return scores,current\n        except RuntimeError as exc:\n            if \'out of memory\' not in str(exc).lower() or current<=1:raise\n            current=max(1,current//2);gc.collect();torch.cuda.empty_cache()\n            progress(f\'COMET GPU OOM: retrying the same uncommitted chunk at batch size {current}\')\n\ndef token_audit(model,records):\n    limit=int(model.encoder.max_positions)-2\n    output={}\n    for role,name in [(\'src\',\'source\'),(\'mt\',\'hypothesis\'),(\'ref\',\'reference\')]:\n        tokens=model.encoder.tokenizer([r[role] for r in records],truncation=False,padding=False,add_special_tokens=True)[\'input_ids\']\n        output[name+\'_token_length\']=np.array([len(x) for x in tokens],int)\n        output[name+\'_truncated\']=output[name+\'_token_length\']>limit\n    return output,limit\n\ndef run(config):\n    require(sha_file(__file__)==config[\'worker_sha256\'],\'Worker source differs from the frozen notebook version\')\n    import torch\n    from huggingface_hub import snapshot_download\n    from comet.models.regression.regression_metric import RegressionMetric\n    require(torch.cuda.is_available(),\'COMET labels need the L4 GPU. Select a GPU runtime; no 60,000-row CPU fallback is started.\')\n    torch.manual_seed(config[\'seed\']);np.random.seed(config[\'seed\'])\n    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False\n    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True\n    torch.set_float32_matmul_precision(\'highest\')\n    output=Path(config[\'output\']);output.mkdir(parents=True,exist_ok=True)\n    wide=load_original_texts(config)\n    input_frames={}\n    for study,info in config[\'inputs\'].items():\n        require(sha_file(info[\'path\'])==info[\'sha256\'],f\'{study}: staged input changed\')\n        frame=pd.read_csv(info[\'path\'],dtype={\'corpus_sentence_id\':str},keep_default_na=False)\n        require(len(frame)==30000 and not frame.duplicated(KEY).any(),\'Incorrect COMET input panel\')\n        # Check every source/reference/hypothesis identity before any new metric score.\n        prepare_records(frame,wide)\n        input_frames[study]=frame\n        progress(f\'{study}: 30,000 original source/reference/hypothesis identities verified\')\n    spec=config[\'metric\']\n    metric_directory=Path(config[\'metric_cache\'])/(\'COMET_\'+spec[\'revision\'][:12])\n    encoder_directory=Path(config[\'metric_cache\'])/(\'XLMR_assets_\'+spec[\'encoder_revision\'][:12])\n    model_dir=snapshot_download(repo_id=spec[\'repo\'],revision=spec[\'revision\'],local_dir=str(metric_directory),\n        allow_patterns=[\'checkpoints/model.ckpt\',\'hparams.yaml\'])\n    encoder_dir=snapshot_download(repo_id=spec[\'encoder_repo\'],revision=spec[\'encoder_revision\'],local_dir=str(encoder_directory),\n        allow_patterns=[\'config.json\',\'tokenizer.json\',\'tokenizer_config.json\',\'special_tokens_map.json\',\'sentencepiece.bpe.model\'])\n    checkpoint=Path(model_dir)/\'checkpoints/model.ckpt\'\n    require(sha_file(checkpoint)==spec[\'checkpoint_sha256\'],\'Pinned COMET checkpoint checksum mismatch\')\n    # The official COMET loader uses this class with load_pretrained_weights=False.\n    # Override only the encoder asset path so tokenizer/config are pinned locally;\n    # the full model weights, including the encoder, come from the COMET checkpoint.\n    model=RegressionMetric.load_from_checkpoint(str(checkpoint),map_location=\'cpu\',strict=False,\n        load_pretrained_weights=False,local_files_only=True,pretrained_model=str(encoder_dir))\n    model.eval();model.float()\n    package_names=[\'unbabel-comet\',\'torch\',\'transformers\',\'numpy\',\'pandas\',\'pytorch-lightning\',\'torchmetrics\',\'huggingface-hub\']\n    environment={p:importlib.metadata.version(p) for p in package_names}\n    signature={**spec,\'score_precision\':\'fp32\',\'allow_tf32\':False,\'packages\':environment,\n        \'worker_sha256\':config[\'worker_sha256\'],\n        \'encoder_assets_sha256\':{p.name:sha_file(p) for p in Path(encoder_dir).iterdir() if p.is_file() and p.suffix in {\'.json\',\'.model\'}},\n        \'encoder_asset_path_override\':\'pinned local config/tokenizer; checkpoint encoder weights retained\',\n        \'gpu\':torch.cuda.get_device_name(0),\'seed\':config[\'seed\'],\n        \'attention_implementation\':str(getattr(model.encoder.model.config,\'_attn_implementation\',\'unspecified\'))}\n    freeze(output/\'metric_run_lock.json\',signature)\n    batch_size=config[\'batch_size\']\n    for study,frame in input_frames.items():\n        folder=output/study;folder.mkdir(parents=True,exist_ok=True)\n        fingerprint=digest(json.dumps({\'input_sha256\':config[\'inputs\'][study][\'sha256\'],\'metric\':signature},sort_keys=True))\n        freeze(folder/\'score_lock.json\',{\'fingerprint\':fingerprint,\'input\':config[\'inputs\'][study],\'metric\':signature})\n        final=folder/\'comet_scores.csv\';complete=folder/\'complete.json\'\n        if complete.exists():\n            old=json.loads(complete.read_text())\n            require(old[\'fingerprint\']==fingerprint and old[\'score_sha256\']==sha_file(final),\'Existing COMET completion does not match\')\n            progress(f\'{study}: verified completed COMET scores; skipping inference\');continue\n        parts=folder/\'parts\';parts.mkdir(exist_ok=True)\n        cached=[]\n        next_part=0\n        for path in sorted(parts.glob(\'part_*.csv\')):\n            part=pd.read_csv(path,dtype={\'corpus_sentence_id\':str});require(part.fingerprint.eq(fingerprint).all(),\'Part cache signature mismatch\')\n            require(np.isfinite(part.comet).all(),\'Nonfinite cached COMET score\')\n            match=part.merge(frame[KEY+[\'hyp_sha256\',\'source_sha256\',\'target_sha256\']],on=KEY,validate=\'one_to_one\',suffixes=(\'\',\'_input\'))\n            require(len(match)==len(part),\'Cached rows do not belong to this input\')\n            for column in [\'hyp_sha256\',\'source_sha256\',\'target_sha256\']:\n                require(match[column].eq(match[column+\'_input\']).all(),\'Cached COMET text identity differs\')\n            cached.append(part)\n            next_part=max(next_part,int(path.stem.split(\'_\')[-1])+1)\n        existing=pd.concat(cached,ignore_index=True) if cached else pd.DataFrame(columns=KEY)\n        require(not existing.duplicated(KEY).any(),\'Duplicate cached COMET rows\')\n        expected_keys=set(map(tuple,frame[KEY].values));done=set(map(tuple,existing[KEY].values))\n        require(done<=expected_keys,\'Cached COMET keys do not belong to this input\')\n        remaining=frame[[tuple(row) not in done for row in frame[KEY].values]].reset_index(drop=True)\n        if len(remaining):\n            smoke=remaining.iloc[:32];records=prepare_records(smoke,wide)\n            start=time.perf_counter();a,batch_size=score_adaptive(model,records,batch_size)\n            torch.cuda.synchronize();seconds=time.perf_counter()-start\n            require(float(np.std(a))>1e-6,\'COMET smoke scores are effectively constant; stop before full scoring and inspect model loading\')\n            b,_=score_adaptive(model,records,min(batch_size,4))\n            require(np.max(np.abs(a-b))<2e-5,\'COMET smoke batch-invariance check failed\')\n            atomic_json(folder/\'smoke_timing.json\',{\'rows\':len(smoke),\'seconds\':seconds,\'rows_per_second\':len(smoke)/seconds,\n                \'remaining_rows\':len(remaining),\'rough_remaining_minutes\':seconds/len(smoke)*len(remaining)/60,\n                \'batch_size_used\':batch_size,\'maximum_batch_comparison_delta\':float(np.max(np.abs(a-b))),\n                \'smoke_score_std\':float(np.std(a)),\n                \'estimate_excludes\':\'environment setup, model downloads, chunk/Drive overhead; first batch can be slower\'})\n            progress(f\'{study}: smoke passed; {len(smoke)/seconds:.1f} rows/s; rough remaining compute {seconds/len(smoke)*len(remaining)/60:.1f} min\')\n        started=time.perf_counter()\n        for start in range(0,len(remaining),config[\'chunk_rows\']):\n            chunk=remaining.iloc[start:start+config[\'chunk_rows\']]\n            records=prepare_records(chunk,wide);audits,limit=token_audit(model,records)\n            scores,batch_size=score_adaptive(model,records,batch_size)\n            part=chunk[KEY+[\'hyp_sha256\',\'source_sha256\',\'target_sha256\']].copy()\n            part[\'comet\']=scores;part[\'fingerprint\']=fingerprint;part[\'batch_size_used\']=batch_size\n            for key,value in audits.items():part[key]=value\n            part[\'metric_token_limit\']=limit\n            part_path=parts/f\'part_{next_part:05d}.csv\';atomic_csv(part_path,part);cached.append(part);next_part+=1\n            progress(f\'{study}: COMET {len(existing)+min(start+len(chunk),len(remaining)):,}/30,000; {time.perf_counter()-started:.1f}s this session\')\n        scored=pd.concat(cached,ignore_index=True)\n        require(len(scored)==30000 and not scored.duplicated(KEY).any(),\'Incomplete final COMET labels\')\n        scored=scored.sort_values([\'pair\',\'corpus_sentence_id\'],key=lambda s:s.astype(int) if s.name==\'corpus_sentence_id\' else s)\n        atomic_csv(final,scored)\n        counts={k:int(scored[k].astype(str).str.lower().isin([\'true\',\'1\',\'1.0\']).sum()) for k in [\'source_truncated\',\'hypothesis_truncated\',\'reference_truncated\']}\n        atomic_json(complete,{\'fingerprint\':fingerprint,\'metric_repo\':spec[\'repo\'],\'metric_revision\':spec[\'revision\'],\n            \'worker_sha256\':config[\'worker_sha256\'],\n            \'encoder_revision\':spec[\'encoder_revision\'],\'score_sha256\':sha_file(final),\'input_sha256\':config[\'inputs\'][study][\'sha256\'],\n            \'rows\':30000,\'truncation_counts\':counts,\'metric_token_limit\':int(scored.metric_token_limit.iloc[0])})\n        progress(f\'{study}: COMET complete; truncation counts {counts}\')\n    atomic_json(output/\'all_complete.json\',{\'studies\':list(input_frames),\'metric_repo\':spec[\'repo\'],\'metric_revision\':spec[\'revision\'],\n        \'completion_sha256\':{s:sha_file(output/s/\'complete.json\') for s in input_frames}})\n\nif __name__==\'__main__\':\n    parser=argparse.ArgumentParser();parser.add_argument(\'--config\',type=Path,required=True)\n    args=parser.parse_args();run(json.loads(args.config.read_text()))\n'
COMET_ENVIRONMENT_LOCK='aiohappyeyeballs==2.7.1\naiohttp==3.14.5\naiosignal==1.4.0\nattrs==26.1.0\ncertifi==2026.10.23\ncharset-normalizer==3.5.2\ncolorama==0.4.6\ndatasets==4.0.0\ndill==0.3.8\nentmax==1.3\nfilelock==3.32.7\nfrozenlist==1.8.0\nfsspec==2025.3.0\nhuggingface-hub==0.30.2\nidna==3.20\njinja2==3.1.6\njsonargparse==3.13.1\nlightning-utilities==0.15.3\nlxml==6.1.3\nmarkupsafe==3.0.4\nmpmath==1.3.0\nmultidict==7.2.1\nmultiprocess==0.70.16\nnetworkx==3.7\nnumpy==1.26.4\nnvidia-cublas-cu12==12.1.3.1\nnvidia-cuda-cupti-cu12==12.1.105\nnvidia-cuda-nvrtc-cu12==12.1.105\nnvidia-cuda-runtime-cu12==12.1.105\nnvidia-cudnn-cu12==9.1.0.70\nnvidia-cufft-cu12==11.0.2.54\nnvidia-curand-cu12==10.3.2.106\nnvidia-cusolver-cu12==11.4.5.107\nnvidia-cusparse-cu12==12.1.0.106\nnvidia-nccl-cu12==2.21.5\nnvidia-nvjitlink-cu12==12.1.105\nnvidia-nvtx-cu12==12.1.105\npackaging==26.3\npandas==2.2.3\nportalocker==4.4.2\npropcache==0.5.4\nprotobuf==4.25.3\npyarrow==20.0.0\npython-dateutil==2.9.0.post0\npytorch-lightning==2.4.0\npytz==2026.6\npyyaml==6.0.3\nregex==2026.9.29\nrequests==2.34.2\nsacrebleu==2.3.1\nsafetensors==0.8.0\nscipy==1.13.1\nsentencepiece==0.2.0\nsetuptools==75.8.0\nsix==1.17.0\nsympy==1.13.1\ntabulate==0.10.0\ntokenizers==0.21.1\ntorch==2.5.1+cu121\ntorchmetrics==0.10.3\ntqdm==4.70.1\ntransformers==4.51.3\ntriton==3.1.0\ntyping-extensions==4.16.0\ntzdata==2026.6\nunbabel-comet==2.2.7\nurllib3==2.8.0\nxxhash==4.0.2\nyarl==1.25.1\n'
freeze_json(OUTPUT_ROOT/'protocol.json',FROZEN_PROTOCOL)
freeze_json(OUTPUT_ROOT/'meaning_split.json',FROZEN_SPLIT)
freeze_json(OUTPUT_ROOT/'expected_input_hashes.json',EXPECTED_INPUTS)
freeze_json(OUTPUT_ROOT/'metric_spec.json',METRIC_SPEC)
freeze_json(OUTPUT_ROOT/'analysis_settings.json',{'core_sha256':CORE_SHA256,'worker_sha256':WORKER_SHA256,
    'n_boot':N_BOOT,'bootstrap_seed':BOOT_SEED,'bootstrap_batch_size':BOOT_BATCH_SIZE,
    'quality_models':METHODS,'primary_outcome':'COMET22-DA','secondary_outcome':'chrF++',
    'confidence_protocol':'retokenized-saved-hypothesis-content-logprob-v2'})
atomic_json(OUTPUT_ROOT/'host_environment.json',{'python':platform.python_version(),'numpy':np.__version__,
    'pandas':pd.__version__,'scoring_in_separate_environment':True})
try:
    from threadpoolctl import threadpool_limits
    _cpu_thread_limit=threadpool_limits(limits=1)
except ImportError:
    pass
print('E2 private protocol, metric revisions, feature sets and 525/112/113 meaning split frozen.')


# %% original code cell 7; notebook cell 13
# 3. Load the completed historical generation/quality/likelihood panels.
FRAMES={};INPUT_PROVENANCE={}
for study in ['it2','nllb']:
    progress(f'Validating existing {study} inputs...')
    FRAMES[study],INPUT_PROVENANCE[study]=load_existing(study,BACKUP_ROOT,CONFIDENCE_ROOT,EXPECTED_INPUTS)
    atomic_json(OUTPUT_ROOT/study/'input_provenance.json',INPUT_PROVENANCE[study])
    progress(f'{study}: 30,000 output rows verified; {INPUT_PROVENANCE[study]["eligible_attention_rows"]:,} attention-eligible')
assert sorted(FRAMES['it2'].corpus_sentence_id.unique(),key=int)==sorted(FRAMES['nllb'].corpus_sentence_id.unique(),key=int)
assert make_meaning_split(FRAMES['it2'].corpus_sentence_id.unique(),FROZEN_SPLIT['seed'])==FROZEN_SPLIT
SCORING_INPUTS=write_scoring_inputs(FRAMES,STAGING)
coverage=[]
for study,d in FRAMES.items():
    for partition,ids in [('train',FROZEN_SPLIT['train']),('validation',FROZEN_SPLIT['validation']),('test',FROZEN_SPLIT['test'])]:
        g=d[d.corpus_sentence_id.isin(ids)]
        coverage.append({'study':study,'partition':partition,'meanings':g.corpus_sentence_id.nunique(),
            'all_quality_rows':len(g),'eligible_attention_rows':int(g.eligible_attention.sum()),
            'eligible_attention_meanings':g.loc[g.eligible_attention,'corpus_sentence_id'].nunique(),
            'excluded_attention_rows':int((~g.eligible_attention).sum()),'cap_hits':int(g.generation_capped.sum())})
atomic_csv(OUTPUT_ROOT/'attention_coverage.csv',pd.DataFrame(coverage));display(pd.DataFrame(coverage))


# %% original code cell 8; notebook cell 15
# 4. Prepare worker configuration; skip all GPU setup if both label caches are complete.
SCORE_OUTPUT=OUTPUT_ROOT/'comet_scores'
TEXT_CACHE=OUTPUT_ROOT/'private_text_cache/flores_source_texts.csv'
TEXT_CANDIDATES=[str(TEXT_CACHE),
    '/content/strong_downstream_inputs/flores_source_texts.csv',
    '/content/drive/MyDrive/indic_40_downstream_strong/flores_source_texts.csv']
WORKER_CONFIG={'seed':20261103,'metric':METRIC_SPEC,'inputs':SCORING_INPUTS,'worker_sha256':WORKER_SHA256,
    'output':str(SCORE_OUTPUT),'metric_cache':str(METRIC_CACHE),'text_cache':str(TEXT_CACHE),
    'existing_text_caches':TEXT_CANDIDATES[1:],'dataset':'openlanguagedata/flores_plus',
    'dataset_revision':'5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06',
    'batch_size':COMET_BATCH_SIZE,'chunk_rows':COMET_CHUNK_ROWS}
atomic_json(STAGING/'worker_config.json',WORKER_CONFIG)
(STAGING/'comet_worker.py').write_text(WORKER_SOURCE,encoding='utf-8')
assert sha_file(STAGING/'comet_worker.py')==WORKER_SHA256
(STAGING/'comet_environment_lock.txt').write_text(COMET_ENVIRONMENT_LOCK,encoding='utf-8')
def completed_labels_match(study):
    path=SCORE_OUTPUT/study/'complete.json';scores=SCORE_OUTPUT/study/'comet_scores.csv'
    if not path.is_file() or not scores.is_file():return False
    info=json.loads(path.read_text())
    return (info.get('metric_revision')==METRIC_SPEC['revision'] and info.get('metric_repo')==METRIC_SPEC['repo']
        and info.get('worker_sha256')==WORKER_SHA256
        and info.get('input_sha256')==SCORING_INPUTS[study]['sha256'] and info.get('score_sha256')==sha_file(scores))
LABELS_READY=all(completed_labels_match(s) for s in FRAMES)
worker_environment=os.environ.copy();worker_environment.pop('PYTHONPATH',None)
worker_environment['PYTHONNOUSERSITE']='1';worker_environment['HF_HUB_DISABLE_XET']='1'
if not LABELS_READY:
    gpu_check=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True)
    require(gpu_check.returncode==0,'Select an L4 GPU runtime before COMET scoring.')
    print('Scoring GPU:',gpu_check.stdout.strip())
    token=os.environ.get('HF_TOKEN')
    try:
        from google.colab import userdata
        token=userdata.get('HF_TOKEN') or token
    except Exception:
        pass
    if token:worker_environment['HF_TOKEN']=token
    require(any(Path(p).is_file() for p in TEXT_CANDIDATES) or bool(token),
        'No saved FLORES text cache found. Enable the approved HF_TOKEN Colab secret before installing the scoring environment.')
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','uv==0.9.7'])
    UV=[sys.executable,'-m','uv']
    WORKER_PYTHON=WORKER_ENV_ROOT/'bin/python'
    if not WORKER_PYTHON.is_file():
        subprocess.check_call(UV+['python','install','3.12'])
        subprocess.check_call(UV+['venv','--python','3.12',str(WORKER_ENV_ROOT)])
    progress('Installing/verifying the isolated pinned COMET environment...')
    subprocess.check_call(UV+['pip','install','--python',str(WORKER_PYTHON),
        '--extra-index-url','https://download.pytorch.org/whl/cu121','--index-strategy','unsafe-best-match',
        '-r',str(STAGING/'comet_environment_lock.txt')])
    subprocess.check_call([str(WORKER_PYTHON),'-c',
        'import torch,numpy,comet; assert torch.cuda.is_available(); print("Isolated COMET ready | torch",torch.__version__,"| numpy",numpy.__version__,"| GPU",torch.cuda.get_device_name(0))'],env=worker_environment)
else:
    print('Both completed COMET label caches verified: GPU setup and scoring will be skipped.')


# %% original code cell 9; notebook cell 16
# 5. New primary COMET labels on saved translations. Stream progress; checkpoint to Drive.
if not LABELS_READY:
    progress('Starting pinned COMET worker. The 32-row smoke test prints a throughput estimate before full scoring.')
    import queue,threading
    worker_lines=queue.Queue()
    worker_command=[str(WORKER_PYTHON),'-u',str(STAGING/'comet_worker.py'),
        '--config',str(STAGING/'worker_config.json')]
    worker_log=STAGING/'comet_worker.log'
    print('Live worker log:',worker_log,flush=True)
    worker_process=subprocess.Popen(worker_command,env=worker_environment,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace')
    def forward_worker_output():
        try:
            for line in worker_process.stdout:worker_lines.put(line)
        finally:
            worker_lines.put(None)
    threading.Thread(target=forward_worker_output,daemon=True).start()
    worker_started=time.monotonic();last_heartbeat=worker_started
    try:
        with worker_log.open('a',encoding='utf-8') as log:
            while True:
                try:
                    line=worker_lines.get(timeout=1)
                    if line is None:break
                    print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:
                    pass
                if time.monotonic()-last_heartbeat>=30:
                    progress(f'Worker process running for {(time.monotonic()-worker_started)/60:.1f} min; awaiting completion. This heartbeat does not prove forward progress.')
                    last_heartbeat=time.monotonic()
        worker_returncode=worker_process.wait()
        if worker_returncode:raise subprocess.CalledProcessError(worker_returncode,worker_command)
    except BaseException:
        if worker_process.poll() is None:
            worker_process.terminate()
            try:worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_process.kill();worker_process.wait()
        raise
for study in FRAMES:
    require(completed_labels_match(study),f'{study}: COMET cache did not finish or verify')
    FRAMES[study]=attach_comet(FRAMES[study],SCORE_OUTPUT/study/'comet_scores.csv',
        SCORE_OUTPUT/study/'complete.json',METRIC_SPEC)
    progress(f'{study}: complete COMET labels attached by exact row/text identity')
metric_lock=json.loads((SCORE_OUTPUT/'metric_run_lock.json').read_text())
truncation=[]
for study,d in FRAMES.items():
    for column in ['source_truncated','hypothesis_truncated','reference_truncated']:
        truncation.append({'study':study,'flag':column,'all_output_rows':len(d),'flagged_rows':int(truth(d[column]).sum())})
atomic_csv(OUTPUT_ROOT/'metric_truncation_counts.csv',pd.DataFrame(truncation));display(pd.DataFrame(truncation))


# %% original code cell 10; notebook cell 18
# 6. Held-out regressions and 10,000 paired test-meaning bootstraps, both systems.
summaries=[];contrasts=[];chains=[]
for study,frame in FRAMES.items():
    progress(f'=== E2 CPU ANALYSIS {study.upper()} ===')
    summary,contrast=run_quality_study(frame,FROZEN_SPLIT,OUTPUT_ROOT,N_BOOT,BOOT_SEED,
        BOOT_BATCH_SIZE,CORE_SHA256,{'metric':METRIC_SPEC,'score_file_sha256':sha_file(SCORE_OUTPUT/study/'comet_scores.csv')})
    summaries.append(summary);contrasts.append(contrast)
    for protocol_name in PROTOCOLS:
        stamp=json.loads((OUTPUT_ROOT/study/protocol_name/'completed.json').read_text())
        assert stamp['bootstrap_completed']==N_BOOT
        chains.append(stamp['draw_chain'])
assert len(set(chains))==1,'Test-meaning bootstrap draws were not aligned across methods/protocols/systems.'
SUMMARY=pd.concat(summaries,ignore_index=True);CONTRASTS=pd.concat(contrasts,ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E2_all_metrics_with_CI.csv',SUMMARY)
atomic_csv(OUTPUT_ROOT/'E2_paired_feature_contrasts.csv',CONTRASTS)
primary=CONTRASTS[(CONTRASTS.protocol=='meaning_and_dyad_held_out')&(CONTRASTS.outcome=='comet')&
    (CONTRASTS.method=='I_controlled_likelihood_signed_residual')&(CONTRASTS.comparator=='G_controlled_likelihood')]
print('Primary COMET added-value comparison; positive = better prediction:')
display(primary[['study','metric','estimate','ci_low','ci_high','positive_means']])
display(SUMMARY[(SUMMARY.partition=='test')&(SUMMARY.outcome=='comet')][
    ['study','protocol','method','Pearson_r','Spearman_rho','MAE','RMSE']])


# %% original code cell 11; notebook cell 20
# 7. Optional human-validation material: validation meanings only; ratings are not performed.
wide_path=next((Path(p) for p in TEXT_CANDIDATES if Path(p).is_file()),None)
if wide_path is None:
    print('No text cache available to the main kernel: optional human-review template skipped; metric/regression results remain saved.')
else:
    wide=pd.read_csv(wide_path,dtype={'id':str},keep_default_na=False).set_index('id')
    blinded=[];key_rows=[];rng=np.random.default_rng(20261105)
    for study,d in FRAMES.items():
        validation=d[d.corpus_sentence_id.isin(FROZEN_SPLIT['validation']) & d.eligible_attention]
        for target,group in validation.groupby('target',sort=True):
            ordered=group.sort_values('comet');bands=np.array_split(np.arange(len(ordered)),3)
            selected=[]
            for band,n in zip(bands,[3,4,3]):selected.extend(rng.choice(band,size=n,replace=False).tolist())
            for _,row in ordered.iloc[selected].iterrows():
                review_id=digest(f'{study}|{row.pair}|{row.corpus_sentence_id}|20261105')[:16]
                blinded.append({'review_id':review_id,'source_language':row.source,'target_language':row.target,
                    'source':str(wide.loc[row.corpus_sentence_id,row.source]),'translation':row.hyp_text,
                    'reference':str(wide.loc[row.corpus_sentence_id,row.target]),
                    'adequacy_0_100':'','fluency_0_100':'','reviewer_notes':''})
                key_rows.append({'review_id':review_id,'study':study,'pair':row.pair,'corpus_sentence_id':row.corpus_sentence_id,
                    'comet':row.comet,'sentence_chrfpp':row.sentence_chrfpp})
    blinded=pd.DataFrame(blinded).sample(frac=1,random_state=20261105)
    atomic_csv(OUTPUT_ROOT/'private_manual_review/blinded_review.csv',blinded)
    atomic_csv(OUTPUT_ROOT/'private_manual_review/review_key.csv',pd.DataFrame(key_rows))
    print('Saved 220 validation-only review cases with blank ratings. Model identity and metric scores are kept in a separate key.')
    print('No human-validation result is claimed; keep metric scores hidden from reviewers.')


# %% original code cell 12; notebook cell 21
# 8. Plot predeclared strict COMET contrasts from the saved tables.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib unavailable: tables are complete; plotting skipped.')
else:
    shown=CONTRASTS[(CONTRASTS.protocol=='meaning_and_dyad_held_out')&(CONTRASTS.outcome=='comet')&
        (CONTRASTS.metric=='Spearman_rho')&(CONTRASTS.comparator=='G_controlled_likelihood')]
    methods=['H_controlled_likelihood_raw_LM','I_controlled_likelihood_signed_residual','J_controlled_likelihood_abs_residual']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,study in zip(axes,['it2','nllb']):
        g=shown[shown.study==study].set_index('method')
        for x,method in enumerate(methods):
            row=g.loc[method];ax.vlines(x,row.ci_low,row.ci_high,color='#3568a1',lw=2)
            ax.scatter(x,row.estimate,color='#3568a1',zorder=3)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Raw LM','Signed residual','Absolute residual'],rotation=15)
        ax.set_title(study.upper());ax.set_ylabel('Δ Spearman versus language + length + likelihood')
    fig.savefig(OUTPUT_ROOT/'E2_COMET_added_value.png',dpi=220)
    fig.savefig(OUTPUT_ROOT/'E2_COMET_added_value.pdf')
    display(fig);plt.close(fig)


# %% original code cell 13; notebook cell 23
# 9. Save completion and a result archive, excluding corpus text and metric weights.
atomic_json(OUTPUT_ROOT/'completed_all.json',{'experiment':'E2','studies':['it2','nllb'],
    'primary_outcome':METRIC_SPEC,'secondary_outcome':'saved sentence chrF++','split':FROZEN_SPLIT,
    'bootstrap_repetitions':N_BOOT,'bootstrap_draw_chain':chains[0],
    'ci_scope':'conditional test-meaning bootstrap; fitted-model/training uncertainty not included',
    'attention_precision':'historical fp16; COMET inference fp32','human_ratings_completed':False,
    'summary_sha256':sha_file(OUTPUT_ROOT/'E2_all_metrics_with_CI.csv'),
    'contrasts_sha256':sha_file(OUTPUT_ROOT/'E2_paired_feature_contrasts.csv'),
    'core_sha256':CORE_SHA256,'worker_sha256':WORKER_SHA256})
(OUTPUT_ROOT/'e2_quality_analysis.py').write_text('"""E2 continuous quality prediction on saved generated-output attention. CPU analysis."""\nimport hashlib\nimport json\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n\nKEY=[\'pair\',\'sentence_id\',\'corpus_sentence_id\']\nMETHODS=[\'A_likelihood\',\'B_raw_LM\',\'C_language_identity\',\'D_signed_residual\',\n         \'E_likelihood_raw_LM\',\'F_likelihood_signed_residual\',\n         \'G_controlled_likelihood\',\'H_controlled_likelihood_raw_LM\',\n         \'I_controlled_likelihood_signed_residual\',\'J_controlled_likelihood_abs_residual\',\n         \'K_controlled_likelihood_abs_raw_deviation\']\nMETRICS=[\'Pearson_r\',\'Spearman_rho\',\'MAE\',\'RMSE\']\nPROTOCOLS=[\'meaning_held_out\',\'meaning_and_dyad_held_out\']\nCONFIDENCE_FEATURES=[\'hyp_mean_logprob\',\'hyp_p10_logprob\',\'hyp_min_logprob\',\'hyp_std_logprob\',\'hyp_sum_logprob\',\'hyp_eos_logprob\']\n\ndef progress(message):print(time.strftime(\'%H:%M:%S\'),message,flush=True)\ndef require(condition,message):\n    if not bool(condition):raise ValueError(message)\ndef sha_file(path):\n    h=hashlib.sha256()\n    with Path(path).open(\'rb\') as stream:\n        for chunk in iter(lambda:stream.read(1024*1024),b\'\'):h.update(chunk)\n    return h.hexdigest()\ndef digest(text):return hashlib.sha256(str(text).encode(\'utf-8\')).hexdigest()\ndef atomic_json(path,obj):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+\'.tmp\')\n    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+\'\\n\',encoding=\'utf-8\');tmp.replace(path)\ndef atomic_csv(path,frame):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+\'.tmp\')\n    frame.to_csv(tmp,index=False);tmp.replace(path)\ndef atomic_npz(path,**arrays):\n    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(\'.tmp.npz\')\n    np.savez_compressed(tmp,**arrays);tmp.replace(path)\ndef freeze_json(path,obj):\n    path=Path(path)\n    if path.exists():require(json.loads(path.read_text())==obj,f\'Frozen settings differ: {path}; use a new version/output directory for changes.\')\n    else:atomic_json(path,obj)\ndef truth(s):return s.astype(str).str.strip().str.lower().isin([\'true\',\'1\',\'1.0\'])\ndef reverse(pair):return \'__\'.join(reversed(pair.split(\'__\')))\n\n# E2_SECTION: INPUTS\ndef load_existing(study,backup_root,score_root,expected):\n    dataset=\'holdout\' if study==\'it2\' else \'nllb\'\n    dirname=\'indic_40_balanced_holdout_750_rows250_999_locked_v4\' if study==\'it2\' else \'nllb_40_confirmatory_holdout_750_final_v6\'\n    root=Path(backup_root)/dirname\n    paths={\'attention\':root/\'free_generation_all_stats.csv\',\n           \'quality\':root/\'manifests/sentence_quality_greedy.csv\',\n           \'confidence\':Path(score_root)/(dataset+\'_logprob.csv\'),\n           \'identity\':root/\'manifests/sample_identity.csv\',\n           \'config\':root/\'manifests/run_config.json\',\'source_lock\':root/\'manifests/source_lock.json\'}\n    for name,path in paths.items():require(path.is_file(),f\'Missing {study} {name}: {path}. Use the completed original files; no model scoring is silently rerun.\')\n    hashes={}\n    for name,path in paths.items():\n        hashes[name]=sha_file(path)\n        if name in expected[study]:require(hashes[name]==expected[study][name],f\'{study}: {name} differs from the verified completed run.\')\n    config=json.loads(paths[\'config\'].read_text())\n    require(int(config[\'n_sentences\'])==750,f\'{study}: expected 750 historical meanings\')\n    require((\'indictrans2\' if study==\'it2\' else \'nllb\') in config[\'model\'].lower(),f\'{study}: wrong original model\')\n    cols=set(KEY+[\'hyp_text\',\'mean\',\'src_len\',\'tgt_len\',\'analysis_view\',\'generation_hit_max_length\',\n                   \'generation_ended_with_eos\',\'attention_inference_eligible\',\'attention_tensor_role\'])\n    raw=pd.read_csv(paths[\'attention\'],usecols=lambda c:c in cols,dtype={\'corpus_sentence_id\':str},keep_default_na=False)\n    q=pd.read_csv(paths[\'quality\'],dtype={\'corpus_sentence_id\':str})\n    confidence=pd.read_csv(paths[\'confidence\'],dtype={\'corpus_sentence_id\':str})\n    identity=pd.read_csv(paths[\'identity\'],dtype={\'corpus_sentence_id\':str})\n    for frame,name in [(raw,\'attention\'),(q,\'quality\'),(confidence,\'confidence\'),(identity,\'identity\')]:\n        require(set(KEY)<=set(frame),f\'{study}: missing identity columns in {name}\')\n        require(len(frame)==30000 and not frame.duplicated(KEY).any(),f\'{study}: wrong or duplicate {name} row count\')\n        require(frame.pair.nunique()==40 and frame.groupby(\'pair\').corpus_sentence_id.nunique().eq(750).all(),f\'{study}: incomplete {name} panel\')\n        require(set(frame.corpus_sentence_id)==set(map(str,range(250,1000))),f\'{study}: {name} corpus identities differ\')\n    required_scores=CONFIDENCE_FEATURES+[\'hyp_scored_tokens\',\'source_token_match\',\'hyp_retokenization_match\']\n    require(set(required_scores)<=set(confidence),f\'{study}: incomplete saved v2 likelihood columns\')\n    require(truth(confidence.source_token_match).all(),f\'{study}: confidence source replay mismatch\')\n    require((confidence.hyp_scored_tokens>0).all(),f\'{study}: empty scored hypotheses need an explicit new policy\')\n    score_cols=KEY+required_scores\n    d=raw.merge(q[KEY+[\'sentence_chrfpp\']],on=KEY,validate=\'one_to_one\').merge(confidence[score_cols],on=KEY,validate=\'one_to_one\')\n    d=d.merge(identity[KEY+[\'source_sha256\',\'target_sha256\']],on=KEY,validate=\'one_to_one\')\n    require(len(d)==30000,f\'{study}: row identities do not match across inputs\')\n    require(d.sentence_chrfpp.between(0,100).all(),f\'{study}: invalid chrF++ values\')\n    for col in CONFIDENCE_FEATURES:\n        if col==\'hyp_eos_logprob\':\n            require(not np.isinf(d[col]).any(),f\'{study}: infinite EOS likelihood\');continue\n        require(np.isfinite(d[col]).all(),f\'{study}: nonfinite saved likelihood feature {col}\')\n    require((d.hyp_mean_logprob<=1e-4).all(),f\'{study}: positive mean log likelihood\')\n    eligible=d.analysis_view.eq(\'lexical_no_punctuation\')\n    capped=truth(d.generation_hit_max_length)\n    if study==\'nllb\':\n        require(int(capped.sum())==57,\'NLLB cap count differs from the verified 57 rows\')\n        require({\'attention_inference_eligible\',\'attention_tensor_role\'}<=set(d),\'NLLB eligibility columns missing\')\n        eligible &= ~capped & truth(d.attention_inference_eligible) & d.attention_tensor_role.eq(\'inferential\')\n    require(int(eligible.sum())=={\'it2\':29997,\'nllb\':29943}[study],f\'{study}: eligible attention support changed\')\n    d[\'eligible_attention\']=eligible;d[\'generation_capped\']=capped\n    d[\'LM\']=np.where(eligible,pd.to_numeric(d[\'mean\'],errors=\'coerce\')*pd.to_numeric(d.src_len,errors=\'coerce\'),np.nan)\n    require(np.isfinite(d.loc[eligible,\'LM\']).all() and d.loc[eligible,\'LM\'].between(0,1+1e-5).all(),f\'{study}: invalid eligible lexical mass\')\n    require((pd.to_numeric(d.loc[eligible,\'src_len\'])>0).all() and (pd.to_numeric(d.loc[eligible,\'tgt_len\'])>0).all(),f\'{study}: invalid lexical lengths\')\n    d[[\'source\',\'target\']]=d.pair.str.split(\'__\',expand=True)\n    require(d.source.nunique()==d.target.nunique()==11,f\'{study}: expected the original 11-language graph\')\n    d[\'study\']=study\n    d=d.sort_values([\'pair\',\'corpus_sentence_id\'],key=lambda s:s.astype(int) if s.name==\'corpus_sentence_id\' else s).reset_index(drop=True)\n    info={\'study\':study,\'dataset\':dataset,\'rows\':len(d),\'eligible_attention_rows\':int(eligible.sum()),\n        \'capped_rows\':int(capped.sum()),\'original_attention_collection_precision\':\'fp16\',\n        \'confidence_protocol\':\'retokenized-saved-hypothesis-content-logprob-v2\',\n        \'confidence_hypothesis_retokenization_match_fraction\':float(truth(d.hyp_retokenization_match).mean()),\n        \'input_sha256\':hashes,\'input_paths\':{k:str(v) for k,v in paths.items()}}\n    return d,info\n\ndef make_meaning_split(ids,seed):\n    ids=sorted(map(str,ids),key=int);require(len(ids)==750,\'E2 expects 750 common meanings\')\n    shuffled=np.random.default_rng(seed).permutation(ids)\n    split={\'seed\':seed,\'train\':sorted(map(str,shuffled[:525]),key=int),\n        \'validation\':sorted(map(str,shuffled[525:637]),key=int),\'test\':sorted(map(str,shuffled[637:]),key=int)}\n    require([len(split[k]) for k in [\'train\',\'validation\',\'test\']]==[525,112,113],\'E2 split sizes changed\')\n    return split\n\ndef write_scoring_inputs(frames,staging):\n    files={}\n    for study,d in frames.items():\n        out=d[KEY+[\'hyp_text\',\'source\',\'target\',\'source_sha256\',\'target_sha256\']].copy()\n        out[\'hyp_sha256\']=out.hyp_text.map(digest)\n        p=Path(staging)/(study+\'_comet_inputs.csv\');atomic_csv(p,out)\n        files[study]={\'path\':str(p),\'sha256\':sha_file(p),\'rows\':len(out)}\n    return files\n\ndef attach_comet(frame,score_path,completion_path,expected_metric):\n    score_path=Path(score_path);completion=json.loads(Path(completion_path).read_text())\n    require(completion[\'metric_repo\']==expected_metric[\'repo\'] and completion[\'metric_revision\']==expected_metric[\'revision\'],\'COMET cache is from a different metric revision\')\n    require(completion[\'score_sha256\']==sha_file(score_path),\'COMET score file changed after completion\')\n    scores=pd.read_csv(score_path,dtype={\'corpus_sentence_id\':str})\n    require(len(scores)==30000 and not scores.duplicated(KEY).any(),\'Incomplete COMET output\')\n    columns=KEY+[\'comet\',\'hyp_sha256\',\'source_sha256\',\'target_sha256\',\'source_truncated\',\'hypothesis_truncated\',\'reference_truncated\']\n    require(set(columns)<=set(scores),\'COMET score/audit columns missing\')\n    merged=frame.merge(scores[columns],on=KEY,validate=\'one_to_one\',suffixes=(\'\',\'_score\'))\n    require(len(merged)==len(frame),\'COMET rows do not match the original hypotheses\')\n    require(merged.hyp_text.map(digest).eq(merged.hyp_sha256).all(),\'COMET scored a different hypothesis text\')\n    for col in [\'source_sha256\',\'target_sha256\']:\n        require(merged[col].eq(merged[col+\'_score\']).all(),\'COMET source/reference identity mismatch\')\n    require(np.isfinite(merged.comet).all(),\'Nonfinite COMET scores; no silent dropping\')\n    return merged\n\n# E2_SECTION: CALIBRATION\ndef language_design(d,languages):\n    n=len(languages);parts=[np.ones((len(d),1))]\n    for role in [\'source\',\'target\']:\n        parts.append(np.column_stack([d[role].eq(l).to_numpy(float) for l in languages])-1/n)\n    return np.column_stack(parts)\n\ndef calibration_fit(training,languages):\n    require(len(training)>0,\'Empty calibration training data\')\n    means=training.groupby([\'pair\',\'source\',\'target\'],sort=True,as_index=False).LM.mean()\n    X=language_design(means,languages);rank=int(np.linalg.matrix_rank(X))\n    require(rank==2*len(languages)-1,f\'Language calibration is not identifiable: rank {rank}\')\n    coefficient=np.linalg.pinv(X)@means.LM.to_numpy(float)\n    return coefficient,rank,means.pair.tolist()\n\ndef crossfit_residuals(training,evaluation,languages,n_folds,seed):\n    """OOF residuals for quality-model training; full-training fit for evaluation."""\n    ids=sorted(training.corpus_sentence_id.unique(),key=int)\n    folds=np.array_split(np.random.default_rng(seed).permutation(ids),n_folds)\n    residual=np.full(len(training),np.nan);fold_audit=[]\n    for fi,held_ids in enumerate(folds):\n        held=training.corpus_sentence_id.isin(set(held_ids)).to_numpy()\n        coefficient,rank,pairs=calibration_fit(training.loc[~held],languages)\n        residual[held]=training.loc[held,\'LM\'].to_numpy()-language_design(training.loc[held],languages)@coefficient\n        fold_audit.append({\'inner_fold\':fi,\'train_meanings\':int(training.loc[~held].corpus_sentence_id.nunique()),\n            \'held_meanings\':len(held_ids),\'rank\':rank,\'held_id_sha256\':digest(\'|\'.join(sorted(map(str,held_ids),key=int)))})\n    require(np.isfinite(residual).all(),\'Cross-fit left unassigned residuals\')\n    coef,rank,pairs=calibration_fit(training,languages)\n    evaluation_residual=evaluation.LM.to_numpy()-language_design(evaluation,languages)@coef\n    fixed_training_residual=training.LM.to_numpy()-language_design(training,languages)@coef\n    return residual,evaluation_residual,fixed_training_residual,coef,rank,pairs,fold_audit\n\ndef features(d,residual,languages,raw_center):\n    L=language_design(d,languages)[:,1:]\n    likelihood=d.hyp_mean_logprob.to_numpy(float)[:,None]\n    raw=d.LM.to_numpy(float)[:,None];r=np.asarray(residual,float)[:,None]\n    lengths=np.column_stack([np.log1p(d.src_len.to_numpy(float)),np.log1p(d.tgt_len.to_numpy(float)),\n        np.abs(np.log((d.tgt_len.to_numpy(float)+1)/(d.src_len.to_numpy(float)+1)))])\n    confidence=d[CONFIDENCE_FEATURES].to_numpy(float)\n    controlled=np.column_stack([L,lengths,confidence,np.isnan(confidence).astype(float)])\n    return {\n        \'A_likelihood\':likelihood,\'B_raw_LM\':raw,\'C_language_identity\':L,\'D_signed_residual\':r,\n        \'E_likelihood_raw_LM\':np.column_stack([likelihood,raw]),\n        \'F_likelihood_signed_residual\':np.column_stack([likelihood,r]),\n        \'G_controlled_likelihood\':controlled,\n        \'H_controlled_likelihood_raw_LM\':np.column_stack([controlled,raw]),\n        \'I_controlled_likelihood_signed_residual\':np.column_stack([controlled,r]),\n        \'J_controlled_likelihood_abs_residual\':np.column_stack([controlled,np.abs(r)]),\n        \'K_controlled_likelihood_abs_raw_deviation\':np.column_stack([controlled,np.abs(raw-raw_center)])}\n\ndef fit_linear(train_X,test_X,train_y):\n    """Fixed OLS, training-only standardization; no tuning or test-statistic use."""\n    require(not np.isinf(train_X).any() and not np.isinf(test_X).any() and np.isfinite(train_y).all(),\'Infinite feature or nonfinite outcome\')\n    # Only saved confidence features may be missing. Attention rows were already\n    # required complete. The feature builder supplies confidence missingness flags.\n    prepared=np.array(train_X,float,copy=True);prepared[:,np.isnan(prepared).all(axis=0)]=0.\n    median=np.nanmedian(prepared,axis=0)\n    train_X=np.where(np.isnan(train_X),median,train_X);test_X=np.where(np.isnan(test_X),median,test_X)\n    mu=train_X.mean(0);scale=train_X.std(0);scale=np.where(scale>1e-10,scale,1.)\n    A=np.column_stack([np.ones(len(train_X)),(train_X-mu)/scale])\n    B=np.column_stack([np.ones(len(test_X)),(test_X-mu)/scale])\n    coefficient=np.linalg.lstsq(A,train_y,rcond=None)[0]\n    return B@coefficient,{\'rank\':int(np.linalg.matrix_rank(A)),\'columns\':A.shape[1],\n        \'coefficient\':coefficient.tolist(),\'training_feature_mean\':mu.tolist(),\'training_feature_scale\':scale.tolist(),\n        \'training_imputation_median\':median.tolist(),\'imputation\':\'confidence only; training median, all-missing training column zero; explicit missingness indicators\'}\n\ndef fit_protocol(frame,split,protocol,output,seed):\n    output=Path(output);output.mkdir(parents=True,exist_ok=True)\n    eligible=frame[frame.eligible_attention].copy()\n    training=eligible[eligible.corpus_sentence_id.isin(split[\'train\'])].reset_index(drop=True)\n    evaluation=eligible[eligible.corpus_sentence_id.isin(split[\'validation\']+split[\'test\'])].reset_index(drop=True)\n    require(set(training.corpus_sentence_id).isdisjoint(evaluation.corpus_sentence_id),\'Meaning leakage\')\n    pairs=sorted(frame.pair.unique());languages=sorted(frame.source.unique())\n    require(len(pairs)==40 and all(reverse(p) in pairs for p in pairs),\'Expected the original reciprocal graph\')\n    dyads=[p for p in pairs if p<reverse(p)]\n    folds=[(\'all_directions\',set(pairs))] if protocol==\'meaning_held_out\' else [(p,{p,reverse(p)}) for p in dyads]\n    rows=[];audits=[];equivalence=[];models=[]\n    for fold_index,(held_name,eval_pairs) in enumerate(folds):\n        train=training if protocol==\'meaning_held_out\' else training[~training.pair.isin(eval_pairs)]\n        test=evaluation[evaluation.pair.isin(eval_pairs)]\n        train=train.reset_index(drop=True);test=test.reset_index(drop=True)\n        require(len(train)>0 and len(test)>0,\'Empty evaluation fold\')\n        if protocol!=\'meaning_held_out\':require(not set(train.pair)&eval_pairs,\'Held-dyad leakage\')\n        rtrain,rtest,rfixed,coef,rank,cal_pairs,inner=crossfit_residuals(train,test,languages,5,seed)\n        if protocol!=\'meaning_held_out\':require(not set(cal_pairs)&eval_pairs,\'Calibration used held-dyad rows\')\n        center=float(train.LM.mean())\n        Xtrain=features(train,rtrain,languages,center);Xtest=features(test,rtest,languages,center)\n        ytrain=train[[\'comet\',\'sentence_chrfpp\']].to_numpy(float)\n        prediction=test[KEY+[\'source\',\'target\',\'LM\',\'hyp_mean_logprob\',\'comet\',\'sentence_chrfpp\',\n            \'generation_capped\',\'source_truncated\',\'hypothesis_truncated\',\'reference_truncated\']].copy()\n        prediction[\'residual\']=rtest;prediction[\'abs_residual\']=np.abs(rtest)\n        prediction[\'partition\']=np.where(prediction.corpus_sentence_id.isin(split[\'test\']),\'test\',\'validation\')\n        prediction[\'protocol\']=protocol;prediction[\'fold\']=fold_index;prediction[\'held_dyad\']=held_name\n        for method in METHODS:\n            values,model=fit_linear(Xtrain[method],Xtest[method],ytrain)\n            prediction[\'comet__\'+method]=values[:,0];prediction[\'chrfpp__\'+method]=values[:,1]\n            models.append({\'fold\':fold_index,\'method\':method,**model})\n        # Same fitted additive baseline + language indicators is a linear\n        # reparameterization. This diagnostic uses in-sample training residuals\n        # only to verify algebra; all reported residual methods above use OOF.\n        fixed=features(train,rfixed,languages,center)\n        a,_=fit_linear(fixed[\'H_controlled_likelihood_raw_LM\'],Xtest[\'H_controlled_likelihood_raw_LM\'],ytrain)\n        b,_=fit_linear(fixed[\'I_controlled_likelihood_signed_residual\'],Xtest[\'I_controlled_likelihood_signed_residual\'],ytrain)\n        error=float(np.max(np.abs(a-b)))\n        require(error<1e-6,f\'Linear reparameterization audit failed: {error}\')\n        equivalence.append({\'fold\':fold_index,\'held_dyad\':held_name,\'max_prediction_difference\':error,\n            \'scope\':\'same-fit signed residual with language controls; not an added-information claim\'})\n        for role,names in [(\'validation\',split[\'validation\']),(\'test\',split[\'test\'])]:\n            count=test[test.corpus_sentence_id.isin(names)]\n            audits.append({\'fold\':fold_index,\'held_dyad\':held_name,\'partition\':role,\n                \'training_rows\':len(train),\'training_meanings\':train.corpus_sentence_id.nunique(),\n                \'evaluation_rows\':len(count),\'evaluation_meanings\':count.corpus_sentence_id.nunique(),\n                \'calibration_rank\':rank,\'training_pair_count\':train.pair.nunique(),\n                \'training_meaning_sha256\':digest(\'|\'.join(sorted(train.corpus_sentence_id.unique(),key=int))),\n                \'inner_folds\':json.dumps(inner)})\n        rows.append(prediction)\n        progress(f\'{frame.study.iloc[0]} {protocol}: fold {fold_index+1}/{len(folds)} complete\')\n    predictions=pd.concat(rows,ignore_index=True)\n    require(not predictions.duplicated(KEY).any(),\'Duplicate out-of-fold predictions\')\n    require(len(predictions)==len(evaluation),\'Evaluation coverage mismatch\')\n    atomic_csv(output/\'predictions.csv\',predictions)\n    atomic_csv(output/\'fold_audit.csv\',pd.DataFrame(audits));atomic_csv(output/\'linear_reparameterization_audit.csv\',pd.DataFrame(equivalence))\n    atomic_json(output/\'fitted_quality_models.json\',models)\n    return predictions\n\n# E2_SECTION: METRICS\ndef weighted_correlation(x,y,w):\n    w=np.asarray(w,float);n=w.sum(1);mx=np.sum(w*x,axis=1)/n;my=np.sum(w*y,axis=1)/n\n    dx=x-mx[:,None];dy=y-my[:,None]\n    vx=np.sum(w*dx*dx,axis=1);vy=np.sum(w*dy*dy,axis=1)\n    den=np.sqrt(vx*vy)\n    # Floating summation can leave tiny nonzero variance for constant predictions.\n    tolerance=16*np.finfo(float).eps**2\n    valid=(vx>tolerance*np.maximum(np.sum(w*x*x,axis=1),1.)) & (vy>tolerance*np.maximum(np.sum(w*y*y,axis=1),1.))\n    return np.divide(np.sum(w*dx*dy,axis=1),den,out=np.full(len(n),np.nan),where=valid)\n\ndef rank_structure(values):\n    values=np.asarray(values);order=np.argsort(values,kind=\'stable\');ordered=values[order]\n    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]\n    groups=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(values)]))\n    inverse=np.empty(len(order),int);inverse[order]=np.arange(len(order))\n    return order,starts,groups[inverse]\n\ndef weighted_midranks(weights,structure):\n    order,starts,row_group=structure\n    counts=np.add.reduceat(weights[:,order],starts,axis=1)\n    mid=np.cumsum(counts,axis=1)-counts/2\n    return mid[:,row_group]\n\ndef metric_batch(observed,predicted,weights,y_structure=None,p_structure=None):\n    y=np.asarray(observed,float);p=np.asarray(predicted,float);w=np.asarray(weights,float)\n    total=w.sum(1)\n    y_rank=weighted_midranks(w,y_structure or rank_structure(y))\n    p_rank=weighted_midranks(w,p_structure or rank_structure(p))\n    diff=p-y\n    return np.column_stack([weighted_correlation(p[None,:],y[None,:],w),\n        weighted_correlation(p_rank,y_rank,w),w@np.abs(diff)/total,np.sqrt(w@(diff*diff)/total)])\n\ndef point_tables(predictions):\n    rows=[];dyads=[]\n    for partition,g in predictions.groupby(\'partition\',sort=False):\n        for outcome,column in [(\'comet\',\'comet\'),(\'chrfpp\',\'sentence_chrfpp\')]:\n            for method in METHODS:\n                p=g[outcome+\'__\'+method].to_numpy();y=g[column].to_numpy()\n                values=metric_batch(y,p,np.ones((1,len(g))))[0]\n                rows.append({\'partition\':partition,\'outcome\':outcome,\'method\':method,\'rows\':len(g),\n                    \'meanings\':g.corpus_sentence_id.nunique(),**dict(zip(METRICS,values))})\n                for pair,h in g.groupby(\'pair\'):\n                    v=metric_batch(h[column].to_numpy(),h[outcome+\'__\'+method].to_numpy(),np.ones((1,len(h))))[0]\n                    dyads.append({\'partition\':partition,\'outcome\':outcome,\'method\':method,\'pair\':pair,\'rows\':len(h),**dict(zip(METRICS,v))})\n    return pd.DataFrame(rows),pd.DataFrame(dyads)\n\ndef bootstrap_predictions(test,split,n_boot,seed,batch_size,path,fingerprint):\n    ids=list(split[\'test\']);position={x:i for i,x in enumerate(ids)}\n    row_ids=np.array([position[x] for x in test.corpus_sentence_id],int)\n    keys=[(outcome,method) for outcome in [\'comet\',\'chrfpp\'] for method in METHODS]\n    samples=np.full((n_boot,len(keys),len(METRICS)),np.nan);rng=np.random.default_rng(seed);completed=0;chain=\'\'\n    if Path(path).exists():\n        with np.load(path,allow_pickle=False) as z:\n            require(str(z[\'fingerprint\'].item())==fingerprint,\'Bootstrap cache input/settings differ\')\n            require(z[\'samples\'].shape==samples.shape,\'Bootstrap dimensions changed\')\n            samples[:]=z[\'samples\'];completed=int(z[\'completed\'].item());chain=str(z[\'draw_chain\'].item())\n            rng.bit_generator.state=json.loads(str(z[\'rng_state\'].item()))\n        progress(f\'Resuming evaluation bootstrap {completed:,}/{n_boot:,}\')\n    prepared={}\n    for outcome,method in keys:\n        y=test[\'comet\' if outcome==\'comet\' else \'sentence_chrfpp\'].to_numpy(float)\n        p=test[outcome+\'__\'+method].to_numpy(float)\n        prepared[(outcome,method)]=(y,p,rank_structure(y),rank_structure(p))\n    started=time.perf_counter()\n    for start in range(completed,n_boot,batch_size):\n        stop=min(start+batch_size,n_boot);b=stop-start\n        meaning_weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=b)\n        weights=meaning_weights[:,row_ids]\n        chain=digest(chain+hashlib.sha256(meaning_weights.tobytes()).hexdigest())\n        for ki,key in enumerate(keys):\n            y,p,ys,ps=prepared[key]\n            samples[start:stop,ki]=metric_batch(y,p,weights,ys,ps)\n        if start==completed or stop==n_boot or stop//1000>start//1000:\n            atomic_npz(path,samples=samples,completed=np.array(stop),fingerprint=np.array(fingerprint),\n                rng_state=np.array(json.dumps(rng.bit_generator.state)),draw_chain=np.array(chain),\n                keys=np.array([\'|\'.join(k) for k in keys]),metrics=np.array(METRICS))\n            progress(f\'Evaluation bootstrap {stop:,}/{n_boot:,}; {time.perf_counter()-started:.1f}s this session\')\n    return samples,keys,chain\n\ndef intervals_and_contrasts(point,samples,keys):\n    output=point.copy();contrasts=[]\n    for i,row in output.iterrows():\n        if row.partition!=\'test\':continue\n        ki=keys.index((row.outcome,row.method))\n        for mi,metric in enumerate(METRICS):\n            v=samples[:,ki,mi];v=v[np.isfinite(v)]\n            lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)\n            output.loc[i,metric+\'_ci_low\']=lo;output.loc[i,metric+\'_ci_high\']=hi\n            output.loc[i,metric+\'_bootstrap_valid\']=len(v)\n    comparisons=[(\'F_likelihood_signed_residual\',\'A_likelihood\',\'document comparison\'),\n        (\'E_likelihood_raw_LM\',\'A_likelihood\',\'document raw-attention comparison\'),\n        (\'I_controlled_likelihood_signed_residual\',\'G_controlled_likelihood\',\'primary controlled added-value comparison\'),\n        (\'H_controlled_likelihood_raw_LM\',\'G_controlled_likelihood\',\'matched raw-attention comparison\'),\n        (\'J_controlled_likelihood_abs_residual\',\'G_controlled_likelihood\',\'prespecified secondary absolute-residual comparison\'),\n        (\'J_controlled_likelihood_abs_residual\',\'K_controlled_likelihood_abs_raw_deviation\',\'matched nonlinear-feature comparison\')]\n    for outcome in [\'comet\',\'chrfpp\']:\n        for a,b,role in comparisons:\n            ra=point[(point.partition==\'test\')&(point.outcome==outcome)&(point.method==a)].iloc[0]\n            rb=point[(point.partition==\'test\')&(point.outcome==outcome)&(point.method==b)].iloc[0]\n            for mi,metric in enumerate(METRICS):\n                sign=1 if mi<2 else -1\n                v=sign*(samples[:,keys.index((outcome,a)),mi]-samples[:,keys.index((outcome,b)),mi]);v=v[np.isfinite(v)]\n                lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)\n                contrasts.append({\'outcome\':outcome,\'method\':a,\'comparator\':b,\'role\':role,\'metric\':metric,\n                    \'estimate\':sign*(ra[metric]-rb[metric]),\'ci_low\':lo,\'ci_high\':hi,\'bootstrap_valid\':len(v),\n                    \'positive_means\':\'higher correlation\' if mi<2 else \'lower error\'})\n    return output,pd.DataFrame(contrasts)\n\n# E2_SECTION: RUN\ndef run_quality_study(frame,split,output,n_boot,seed,batch_size,code_hash,metric_signature):\n    study=frame.study.iloc[0];all_summary=[];all_contrasts=[]\n    for protocol in PROTOCOLS:\n        folder=Path(output)/study/protocol;folder.mkdir(parents=True,exist_ok=True)\n        signature={\'study\':study,\'protocol\':protocol,\'code_sha256\':code_hash,\'split\':split,\n            \'n_boot\':n_boot,\'seed\':seed,\'batch_size\':batch_size,\'metric_signature\':metric_signature,\n            \'row_input_digest\':digest(frame[KEY+[\'LM\',\'comet\',\'sentence_chrfpp\',\'hyp_mean_logprob\',\'eligible_attention\']].to_csv(index=False))}\n        fingerprint=digest(json.dumps(signature,sort_keys=True));freeze_json(folder/\'analysis_lock.json\',signature)\n        prediction_path=folder/\'predictions.csv\';fit_stamp=folder/\'fit_complete.json\'\n        if fit_stamp.exists():\n            old=json.loads(fit_stamp.read_text());require(old[\'fingerprint\']==fingerprint,\'Fitted prediction cache differs\')\n            require(old[\'predictions_sha256\']==sha_file(prediction_path),\'Fitted predictions were modified\')\n            predictions=pd.read_csv(prediction_path,dtype={\'corpus_sentence_id\':str})\n            progress(f\'{study} {protocol}: reusing verified fitted predictions\')\n        else:\n            predictions=fit_protocol(frame,split,protocol,folder,split[\'seed\'])\n            atomic_json(fit_stamp,{\'fingerprint\':fingerprint,\'predictions_sha256\':sha_file(prediction_path)})\n        point,by_direction=point_tables(predictions)\n        atomic_csv(folder/\'point_metrics.csv\',point);atomic_csv(folder/\'per_direction_metrics.csv\',by_direction)\n        test=predictions[predictions.partition==\'test\'].reset_index(drop=True)\n        samples,keys,chain=bootstrap_predictions(test,split,n_boot,seed,batch_size,folder/\'bootstrap_checkpoint.npz\',fingerprint)\n        summary,contrasts=intervals_and_contrasts(point,samples,keys)\n        summary[\'study\']=study;summary[\'protocol\']=protocol;contrasts[\'study\']=study;contrasts[\'protocol\']=protocol\n        atomic_csv(folder/\'metrics_with_CI.csv\',summary);atomic_csv(folder/\'paired_contrasts.csv\',contrasts)\n        atomic_json(folder/\'completed.json\',{\'fingerprint\':fingerprint,\'bootstrap_completed\':n_boot,\'draw_chain\':chain,\n            \'ci_scope\':\'test-meaning sampling conditional on fixed fitted models; training-fit uncertainty omitted\',\n            \'result_sha256\':{name:sha_file(folder/name) for name in [\'predictions.csv\',\'metrics_with_CI.csv\',\'paired_contrasts.csv\',\'bootstrap_checkpoint.npz\']}})\n        all_summary.append(summary);all_contrasts.append(contrasts)\n    return pd.concat(all_summary,ignore_index=True),pd.concat(all_contrasts,ignore_index=True)\n',encoding='utf-8')
# The analysis source is already in this notebook; retain the worker/lock as run provenance.
(OUTPUT_ROOT/'comet_worker.py').write_text(WORKER_SOURCE,encoding='utf-8')
(OUTPUT_ROOT/'comet_environment_lock.txt').write_text(COMET_ENVIRONMENT_LOCK,encoding='utf-8')
import zipfile
archive=OUTPUT_ROOT.parent/(OUTPUT_ROOT.name+'_results.zip');tmp=archive.with_suffix('.tmp.zip')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(OUTPUT_ROOT.rglob('*')):
        rel=path.relative_to(OUTPUT_ROOT)
        if path.is_file() and not any(p in {'private_text_cache','private_manual_review','metric_cache'} for p in rel.parts) and not path.name.endswith('.tmp'):
            zipped.write(path,rel.as_posix())
tmp.replace(archive)
print('E2 COMPLETE. Results:',OUTPUT_ROOT)
print('Results archive:',archive)
print('Share E2_all_metrics_with_CI.csv, E2_paired_feature_contrasts.csv and completed_all.json for review.')

