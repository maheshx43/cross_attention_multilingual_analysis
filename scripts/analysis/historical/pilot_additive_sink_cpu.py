# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1
import hashlib, json, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

PRIMARY='pilot'
ROOTS={
 'pilot':Path('/content/drive/MyDrive/indic_40_backups/indic_40_balanced_pilot_250_protocol_v3'),
 'holdout':Path('/content/drive/MyDrive/indic_40_backups/indic_40_balanced_holdout_750_rows250_999_locked_v4'),
 'nllb':Path('/content/drive/MyDrive/indic_40_backups/nllb_40_confirmatory_holdout_750_final_v6'),
}
EXPECTED_N={'pilot':250,'holdout':750,'nllb':750}
EXPECTED_PAIRED={'pilot':{'teacher_forced':5000,'free_generation':4997},
                 'holdout':{'teacher_forced':15000,'free_generation':14997},
                 'nllb':{'teacher_forced':15000,'free_generation':14944}}
MODES=('teacher_forced','free_generation')
N_TENSOR_MEANINGS=100
N_EXACT_BOOT=5000
TENSOR_SAMPLE_SEED=20260929
OUTPUT=Path('/content/drive/MyDrive/indic_40_additive_sink_cpu')/PRIMARY
OUTPUT.mkdir(parents=True,exist_ok=True)
print('CPU only:',PRIMARY,'| exact tensor meanings per mode:',N_TENSOR_MEANINGS,'|',OUTPUT)


# %% original code cell 1; notebook cell 3
def safe_mean(a,axis=0):
    a=np.asarray(a,float);good=np.isfinite(a)
    n=good.sum(axis=axis);total=np.where(good,a,0.0).sum(axis=axis)
    return np.divide(total,n,out=np.full(np.shape(total),np.nan,float),where=n>0)

def pearson_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else np.nan

def spearman_corr(x,y):
    return pearson_corr(pd.Series(x).rank().to_numpy(float),
                        pd.Series(y).rank().to_numpy(float))

def reverse(p):
    a,b=p.split('__');return b+'__'+a

def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(['1','1.0','true','yes'])

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

root=ROOTS[PRIMARY]
assert root.is_dir(),f'Missing original GPU collector backup: {root}'
config=json.loads((root/'manifests'/'run_config.json').read_text(encoding='utf8'))
assert int(config['n_sentences'])==EXPECTED_N[PRIMARY]
assert ('nllb' if PRIMARY=='nllb' else 'indictrans2') in str(config['model']).lower()
manifest=pd.read_csv(root/'manifests'/'sample_identity.csv',dtype={'corpus_sentence_id':str})
assert not manifest.duplicated(['pair','corpus_sentence_id']).any()

frames={};input_hashes={}
for mode in MODES:
    path=root/f'{mode}_all_stats.csv';assert path.is_file(),path
    input_hashes[mode]=sha256_file(path)
    needed={'pair','sentence_id','corpus_sentence_id','mean','src_len','tgt_len',
            'analysis_view','all_retained_mean','all_retained_src_len',
            'source_tokens_json','target_tokens_json','source_tags_removed',
            'src_retained_tokens'}
    if PRIMARY=='nllb' and mode=='free_generation':
        needed|={'generation_hit_max_length','attention_inference_eligible','attention_tensor_role'}
    if PRIMARY=='nllb': needed.add('source_special_tokens_removed')
    df=pd.read_csv(path,usecols=lambda x:x in needed,dtype={'corpus_sentence_id':str})
    assert needed.issubset(df.columns),sorted(needed-set(df.columns))
    assert len(df)==40*EXPECTED_N[PRIMARY] and df.pair.nunique()==40
    assert not df.duplicated(['pair','corpus_sentence_id']).any()
    assert not df.duplicated(['pair','sentence_id']).any()
    assert set(df.analysis_view.dropna().unique()).issubset({
        'lexical_no_punctuation','lexical_no_punctuation_unavailable',
        'excluded_generation_cap'})
    assert (df.source_tags_removed.astype(int)==(1 if PRIMARY=='nllb' else 2)).all()
    if PRIMARY=='nllb': assert (df.source_special_tokens_removed.astype(int)==2).all()
    assert df.groupby('pair').sentence_id.apply(
        lambda s:set(s.astype(int))==set(range(EXPECTED_N[PRIMARY]))).all()
    eligible=df.analysis_view.eq('lexical_no_punctuation')
    if PRIMARY=='nllb' and mode=='free_generation':
        capped=truthy(df.generation_hit_max_length)
        assert int(capped.sum())==57
        eligible &= ~capped & truthy(df.attention_inference_eligible)
        eligible &= df.attention_tensor_role.astype(str).str.lower().eq('inferential')
        assert int(eligible.sum())==29943
    df['eligible']=eligible
    assert np.allclose(df.loc[eligible,'src_retained_tokens'],
                       df.loc[eligible,'all_retained_src_len'])
    lm=pd.to_numeric(df['mean'],errors='coerce')*pd.to_numeric(df['src_len'],errors='coerce')
    retained=pd.to_numeric(df['all_retained_mean'],errors='coerce')*\
             pd.to_numeric(df['all_retained_src_len'],errors='coerce')
    df['lm']=np.where(eligible,lm,np.nan)
    df['retained_proxy']=np.where(eligible,retained,np.nan)
    df['control_sink_proxy']=np.where(eligible,1.0-retained,np.nan)
    assert df.loc[eligible,['lm','retained_proxy','control_sink_proxy']].notna().all().all()
    assert (df.loc[eligible,'retained_proxy'].between(-0.002,1.002)).all()
    assert df.loc[~eligible,['lm','retained_proxy','control_sink_proxy']].isna().all().all()
    frames[mode]=df

pairs=sorted(frames['teacher_forced'].pair.unique())
assert all(sorted(f.pair.unique())==pairs for f in frames.values())
dyads=[p for p in pairs if p<reverse(p)]
assert len(dyads)==20 and all(reverse(p) in pairs for p in dyads)
langs=sorted({x for p in pairs for x in p.split('__')})
assert len(langs)==11
B=np.zeros((20,11))
for j,p in enumerate(dyads):
    a,b=p.split('__');B[j,langs.index(a)]=1;B[j,langs.index(b)]=-1
assert np.linalg.matrix_rank(B)==10
ids=sorted(frames['teacher_forced'].corpus_sentence_id.unique(),key=lambda s:(len(s),s))
assert len(ids)==EXPECTED_N[PRIMARY]
identity=manifest.set_index(['pair','corpus_sentence_id'])
for p in dyads:
    a=identity.loc[p].sort_index();b=identity.loc[reverse(p)].sort_index()
    assert a.index.equals(b.index)
    assert (a.source_sha256.values==b.target_sha256.values).all()
    assert (a.target_sha256.values==b.source_sha256.values).all()

matrices={};edges={};directed={}
for mode,df in frames.items():
    assert all(set(g.corpus_sentence_id)==set(ids) for _,g in df.groupby('pair'))
    matrices[mode]={}
    for metric in ['lm','control_sink_proxy']:
        matrices[mode][metric]=df.pivot(index='corpus_sentence_id',columns='pair',
                                        values=metric).reindex(index=ids,columns=pairs)
    # Use the SAME reciprocal-complete meaning support for both quantities.
    common={}
    for p in dyads:
        q=reverse(p);m=matrices[mode]['lm'];s=matrices[mode]['control_sink_proxy']
        common[p]=(np.isfinite(m[p])&np.isfinite(m[q])&np.isfinite(s[p])&np.isfinite(s[q]))
    edges[mode]={};directed[mode]={}
    for metric in ['lm','control_sink_proxy']:
        pv=matrices[mode][metric]
        edges[mode][metric]=np.stack([
            np.where(common[p],(pv[p]-pv[reverse(p)]).to_numpy(float),np.nan)
            for p in dyads],axis=1)
        matched={}
        for p in dyads:
            q=reverse(p);m=common[p]
            matched[p]=float(pv.loc[m,p].mean());matched[q]=float(pv.loc[m,q].mean())
        directed[mode][metric]=np.array([matched[p] for p in pairs])
    assert int(np.isfinite(edges[mode]['lm']).sum())==EXPECTED_PAIRED[PRIMARY][mode]
print('Paired meaning-edge support:',{m:int(np.isfinite(edges[m]['lm']).sum()) for m in MODES})


# %% original code cell 2; notebook cell 5
def additive_design(pairs,langs,kind='both'):
    n=len(langs);parts=[np.ones((len(pairs),1))]
    if kind in ('both','source'):
        S=np.full((len(pairs),n),-1/n)
        for i,p in enumerate(pairs):S[i,langs.index(p.split('__')[0])]+=1
        parts.append(S)
    if kind in ('both','target'):
        T=np.full((len(pairs),n),-1/n)
        for i,p in enumerate(pairs):T[i,langs.index(p.split('__')[1])]+=1
        parts.append(T)
    return np.concatenate(parts,axis=1)

def r2(y,pred,centered=True):
    y=np.asarray(y,float);pred=np.asarray(pred,float)
    base=np.sum((y-y.mean())**2) if centered else np.sum(y*y)
    return float(1-np.sum((y-pred)**2)/base) if base>0 else np.nan

def directed_loeo(y,kind):
    X=additive_design(pairs,langs,kind) if kind!='mean' else None
    pred=np.full(len(pairs),np.nan);ranks=[]
    for p in dyads:
        test=np.array([pairs.index(p),pairs.index(reverse(p))]);train=np.ones(len(pairs),bool)
        train[test]=False
        if kind=='mean':pred[test]=y[train].mean();continue
        ranks.append(int(np.linalg.matrix_rank(X[train])))
        assert ranks[-1]==np.linalg.matrix_rank(X),f'{kind}: held-out dyad makes design unidentifiable'
        pred[test]=X[test]@(np.linalg.pinv(X[train])@y[train])
    assert np.isfinite(pred).all()
    pair_obs=np.array([(y[pairs.index(p)]+y[pairs.index(reverse(p))])/2 for p in dyads])
    pair_pred=np.array([(pred[pairs.index(p)]+pred[pairs.index(reverse(p))])/2 for p in dyads])
    d_obs=np.array([y[pairs.index(p)]-y[pairs.index(reverse(p))] for p in dyads])
    d_pred=np.array([pred[pairs.index(p)]-pred[pairs.index(reverse(p))] for p in dyads])
    return pred,{'model':kind,'directed_LOEO_R2_centered':r2(y,pred),
                 'directed_LOEO_MAE':float(np.mean(np.abs(y-pred))),
                 'symmetric_pair_mean_LOEO_R2_centered':r2(pair_obs,pair_pred),
                 'reciprocal_difference_LOEO_R2_zero':r2(d_obs,d_pred,False),
                 'n_heldout_dyads':len(dyads),'min_training_rank':min(ranks) if ranks else 1}

loeo_summaries=[];loeo_predictions=[];identity_rows=[]
for mode in MODES:
    y=directed[mode]['lm']
    for kind in ['both','source','target','mean']:
        pred,summary=directed_loeo(y,kind)
        loeo_summaries.append({'mode':mode,**summary})
        loeo_predictions.extend({'mode':mode,'model':kind,'pair':p,
                                 'observed':float(y[j]),'predicted':float(pred[j]),
                                 'error':float(pred[j]-y[j])}
                                for j,p in enumerate(pairs))
    X=additive_design(pairs,langs,'both');coef=np.linalg.pinv(X)@y
    alpha,beta=coef[1:12],coef[12:23]
    theta=np.linalg.pinv(B)@safe_mean(edges[mode]['lm'],0)
    identity_rows.append({'mode':mode,'in_sample_directed_R2_centered':r2(y,X@coef),
                          'additive_design_rank':int(np.linalg.matrix_rank(X)),
                          'max_abs_theta_minus_alpha_plus_beta':
                          float(np.max(np.abs(theta-(alpha-beta))))})

loeo_summary=pd.DataFrame(loeo_summaries)
display(loeo_summary)
display(pd.DataFrame(identity_rows))
loeo_summary.to_csv(OUTPUT/'directed_additive_loeo_summary.csv',index=False)
pd.DataFrame(loeo_predictions).to_csv(OUTPUT/'directed_additive_loeo_predictions.csv',index=False)
pd.DataFrame(identity_rows).to_csv(OUTPUT/'additive_identity_in_sample.csv',index=False)


# %% original code cell 3; notebook cell 7
def field_energy(y):
    g=B@(np.linalg.pinv(B)@y)
    return float(np.dot(g,g)/np.dot(y,y))

def one_predictor_loeo(x,y):
    pred=np.empty(len(y))
    for i in range(len(y)):
        mask=np.arange(len(y))!=i
        X=np.column_stack([np.ones(mask.sum()),x[mask]])
        pred[i]=np.array([1.0,x[i]])@(np.linalg.pinv(X)@y[mask])
    return pred

sink_rows=[];sink_edge_rows=[]
for mode in MODES:
    df=frames[mode]
    eligible=df.eligible
    lm=safe_mean(edges[mode]['lm'],0)
    negative_sink=-safe_mean(edges[mode]['control_sink_proxy'],0)
    assert np.isfinite(lm).all() and np.isfinite(negative_sink).all()
    X=np.column_stack([np.ones(20),negative_sink]);coef=np.linalg.pinv(X)@lm
    sink_pred=one_predictor_loeo(negative_sink,lm)
    sink_rows.append({'mode':mode,'eligible_direction_rows':int(eligible.sum()),
      'mean_lexical_mass':float(df.loc[eligible,'lm'].mean()),
      'mean_removed_control_share_all_retained_target_rows':
          float(df.loc[eligible,'control_sink_proxy'].mean()),
      'source_control_share_q05':float(df.loc[eligible,'control_sink_proxy'].quantile(.05)),
      'source_control_share_q95':float(df.loc[eligible,'control_sink_proxy'].quantile(.95)),
      'edge_corr_LM_vs_negative_sink_pearson':pearson_corr(lm,negative_sink),
      'edge_corr_LM_vs_negative_sink_spearman':spearman_corr(lm,negative_sink),
      'edge_R2_LM_from_sink_in_sample_centered':r2(lm,X@coef),
      'edge_R2_LM_from_sink_LOEO_centered':r2(lm,sink_pred),
      'edge_slope_LM_on_negative_sink':float(coef[1]),
      'LM_gradient_energy':field_energy(lm),
      'negative_sink_gradient_energy':field_energy(negative_sink),
      'theta_LM_vs_negative_sink_spearman':spearman_corr(
          np.linalg.pinv(B)@lm,np.linalg.pinv(B)@negative_sink)})
    sink_edge_rows.extend({'mode':mode,'dyad':p,'d_LM':float(lm[j]),
                           'negative_d_control_sink_proxy':float(negative_sink[j]),
                           'd_LM_minus_negative_sink_proxy':float(lm[j]-negative_sink[j]),
                           'LOEO_sink_regression_prediction':float(sink_pred[j])}
                          for j,p in enumerate(dyads))
sink_summary=pd.DataFrame(sink_rows)
display(sink_summary)
display(pd.DataFrame(sink_edge_rows))
sink_summary.to_csv(OUTPUT/'full_metrics_sink_summary.csv',index=False)
pd.DataFrame(sink_edge_rows).to_csv(OUTPUT/'full_metrics_sink_edges.csv',index=False)


# %% original code cell 4; notebook cell 9
def punctuation_piece(piece):
    text=str(piece).replace('▁','').replace('Ġ','').strip()
    return not text or all(unicodedata.category(ch)[0] in {'P','S','Z','C'} for ch in text)

def exact_tensor_row(row,mode):
    source=json.loads(row.source_tokens_json);target=json.loads(row.target_tokens_json)
    src_lex=np.array([j for j,p in enumerate(source) if not punctuation_piece(p)],int)
    tgt_lex=np.array([j for j,p in enumerate(target) if not punctuation_piece(p)],int)
    assert len(src_lex)==int(row.src_len) and len(tgt_lex)==int(row.tgt_len)
    path=root/mode/row.pair/f'{int(row.sentence_id)}.npy'
    assert path.is_file(),f'Missing saved eligible LHTS tensor: {path}'
    full=np.load(path,mmap_mode='r',allow_pickle=False)
    assert full.ndim==4 and full.shape[-2:]==(len(target),len(source))
    matrix=np.asarray(full,dtype=np.float32).mean(axis=(0,1))
    lexical=float(matrix[np.ix_(tgt_lex,src_lex)].sum(axis=1).mean())
    retained=float(matrix[tgt_lex].sum(axis=1).mean())
    punct=retained-lexical
    removed_control=1.0-retained
    conditional=lexical/retained if retained>0 else np.nan
    return {'pair':row.pair,'sentence_id':int(row.sentence_id),
            'corpus_sentence_id':str(row.corpus_sentence_id),'mode':mode,
            'lm_exact':lexical,'punct_exact':punct,'control_sink_exact':removed_control,
            'conditional_lexical_share_exact':conditional,
            'sink_proxy_from_csv':float(row.control_sink_proxy),
            'lm_from_csv':float(row.lm)}

selected_positions=np.sort(np.random.default_rng(TENSOR_SAMPLE_SEED).choice(
    len(ids),size=N_TENSOR_MEANINGS,replace=False))
selected_ids=[ids[int(i)] for i in selected_positions]
assert len(set(selected_ids))==N_TENSOR_MEANINGS
selection_hash=hashlib.sha256('|'.join(selected_ids).encode()).hexdigest()[:12]
print('Joint tensor-audit meaning IDs:',selected_ids)
exact_tables={};tensor_validation=[]
for mode in MODES:
    df=frames[mode]
    cache=OUTPUT/f'exact_sink_rows_{mode}_{N_TENSOR_MEANINGS}_{input_hashes[mode][:12]}_{selection_hash}.csv'
    if cache.exists():
        done=pd.read_csv(cache,dtype={'corpus_sentence_id':str})
        assert not done.duplicated(['pair','corpus_sentence_id']).any()
        assert set(done.corpus_sentence_id).issubset(selected_ids)
    else:
        done=pd.DataFrame()
    done_keys=set(zip(done.pair,done.corpus_sentence_id)) if len(done) else set()
    new=[]
    for pair_i,p in enumerate(pairs):
        subset=df.loc[(df.pair==p)&df.corpus_sentence_id.isin(selected_ids)&df.eligible]
        for row in subset.itertuples(index=False):
            key=(row.pair,str(row.corpus_sentence_id))
            if key not in done_keys:
                new.append(exact_tensor_row(row,mode));done_keys.add(key)
        if new and ((pair_i+1)%4==0 or pair_i==len(pairs)-1):
            done=pd.concat([done,pd.DataFrame(new)],ignore_index=True);new=[]
            temp=cache.with_suffix('.tmp')
            done.to_csv(temp,index=False);temp.replace(cache)
            print(mode,'checkpointed pairs:',pair_i+1,'/',len(pairs),'rows:',len(done))
    expected=int(df.loc[df.eligible&df.corpus_sentence_id.isin(selected_ids)].shape[0])
    assert len(done)==expected,(mode,len(done),expected)
    assert done[['lm_exact','punct_exact','control_sink_exact',
                 'conditional_lexical_share_exact']].notna().all().all()
    closure=np.max(np.abs(done.lm_exact+done.punct_exact+done.control_sink_exact-1))
    lm_delta=np.max(np.abs(done.lm_exact-done.lm_from_csv))
    assert closure<1e-5 and lm_delta<0.005,(closure,lm_delta)
    exact_tables[mode]=done
    tensor_validation.append({'mode':mode,'selected_meanings':N_TENSOR_MEANINGS,
                              'eligible_tensor_rows':len(done),
                              'capped_or_ineligible_sample_rows':40*N_TENSOR_MEANINGS-len(done),
                              'max_mass_closure_error':float(closure),
                              'max_tensor_vs_CSV_LM_difference':float(lm_delta),
                              'sink_proxy_MAE_on_same_rows':float(np.mean(
                                  np.abs(done.control_sink_exact-done.sink_proxy_from_csv))),
                              'sink_proxy_q95_abs_error_on_same_rows':float(np.quantile(
                                  np.abs(done.control_sink_exact-done.sink_proxy_from_csv),.95))})
display(pd.DataFrame(tensor_validation))
pd.DataFrame(tensor_validation).to_csv(OUTPUT/'exact_tensor_validation.csv',index=False)


# %% original code cell 5; notebook cell 11
exact_rows=[];exact_edges=[];convergence_rows=[]
convergence_order=np.random.default_rng(TENSOR_SAMPLE_SEED+55).permutation(N_TENSOR_MEANINGS)
for mode,df in exact_tables.items():
    pvs={name:df.pivot(index='corpus_sentence_id',columns='pair',values=name)
         .reindex(index=selected_ids,columns=pairs)
         for name in ['lm_exact','punct_exact','control_sink_exact','conditional_lexical_share_exact']}
    E={name:np.stack([(pv[p]-pv[reverse(p)]).to_numpy(float)
                      for p in dyads],axis=1) for name,pv in pvs.items()}
    paired_per_dyad=np.isfinite(E['lm_exact']).sum(axis=0)
    minimum_paired=max(20,int(np.ceil(0.6*N_TENSOR_MEANINGS)))
    print(mode,'exact paired meanings per dyad:',paired_per_dyad.tolist(),
          '| required minimum:',minimum_paired)
    assert (paired_per_dyad>=minimum_paired).all(),(mode,paired_per_dyad,minimum_paired)
    # The same two directions and same meanings support all components.
    assert np.array_equal(np.isfinite(E['lm_exact']),np.isfinite(E['punct_exact']))
    assert np.array_equal(np.isfinite(E['lm_exact']),np.isfinite(E['control_sink_exact']))
    assert np.allclose(E['lm_exact'],-E['punct_exact']-E['control_sink_exact'],
                       equal_nan=True,atol=1e-5)
    d={name:safe_mean(value,0) for name,value in E.items()}
    assert all(np.isfinite(v).all() for v in d.values())
    lm=d['lm_exact'];sink=-d['control_sink_exact'];punct=-d['punct_exact'];conditional=d['conditional_lexical_share_exact']
    assert np.allclose(lm,sink+punct,atol=1e-5)
    rng=np.random.default_rng(20260929+(0 if mode=='teacher_forced' else 1))
    boot_energy=np.empty((N_EXACT_BOOT,2))
    for rep in range(N_EXACT_BOOT):
        draw=rng.integers(0,N_TENSOR_MEANINGS,N_TENSOR_MEANINGS)
        boot_energy[rep,0]=field_energy(safe_mean(E['lm_exact'][draw],0))
        boot_energy[rep,1]=field_energy(safe_mean(E['conditional_lexical_share_exact'][draw],0))
    boot_delta=boot_energy[:,1]-boot_energy[:,0]
    for n in [32,64,N_TENSOR_MEANINGS]:
        take=np.sort(convergence_order[:n])
        lm_n=safe_mean(E['lm_exact'][take],0)
        cond_n=safe_mean(E['conditional_lexical_share_exact'][take],0)
        sink_n=-safe_mean(E['control_sink_exact'][take],0)
        assert np.isfinite(lm_n).all() and np.isfinite(cond_n).all()
        convergence_rows.append({'mode':mode,'meanings':len(take),
            'LM_gradient_energy':field_energy(lm_n),
            'conditional_gradient_energy':field_energy(cond_n),
            'energy_change_conditional_minus_LM':field_energy(cond_n)-field_energy(lm_n),
            'sink_to_LM_edge_corr':pearson_corr(lm_n,sink_n)})
    denom=float(np.dot(lm,lm))
    sink_er=float(np.dot(sink,sink)/denom)
    punct_er=float(np.dot(punct,punct)/denom)
    cross_er=float(2*np.dot(sink,punct)/denom)
    assert np.isclose(sink_er+punct_er+cross_er,1.0,atol=1e-5)
    X=np.column_stack([np.ones(20),sink]);coef=np.linalg.pinv(X)@lm
    sink_loeo=one_predictor_loeo(sink,lm)
    exact_rows.append({'mode':mode,'selected_meanings':N_TENSOR_MEANINGS,
       'paired_meaning_edge_observations':int(np.isfinite(E['lm_exact']).sum()),
       'mean_exact_lexical_share':float(df.lm_exact.mean()),
       'mean_exact_punctuation_share':float(df.punct_exact.mean()),
       'mean_exact_removed_control_share':float(df.control_sink_exact.mean()),
       'edge_R2_LM_from_negative_sink_centered':r2(lm,X@coef),
       'edge_R2_LM_from_negative_sink_LOEO_centered':r2(lm,sink_loeo),
       'minimum_paired_meanings_required':minimum_paired,
       'minimum_paired_meanings_observed':int(paired_per_dyad.min()),
       'sink_component_energy_ratio':sink_er,
       'punct_component_energy_ratio':punct_er,
       'cross_component_energy_ratio':cross_er,
       'LM_gradient_energy':field_energy(lm),
       'conditional_lexical_share_gradient_energy':field_energy(conditional),
       'LM_gradient_energy_boot_lo':float(np.quantile(boot_energy[:,0],.025)),
       'LM_gradient_energy_boot_hi':float(np.quantile(boot_energy[:,0],.975)),
       'conditional_gradient_energy_boot_lo':float(np.quantile(boot_energy[:,1],.025)),
       'conditional_gradient_energy_boot_hi':float(np.quantile(boot_energy[:,1],.975)),
       'conditional_minus_LM_energy_boot_lo':float(np.quantile(boot_delta,.025)),
       'conditional_minus_LM_energy_boot_hi':float(np.quantile(boot_delta,.975)),
       'LM_edge_RMS':float(np.sqrt(np.mean(lm**2))),
       'conditional_share_edge_RMS':float(np.sqrt(np.mean(conditional**2))),
       'conditional_share_LOEO_R2_zero':r2(conditional,np.array([
           B[j]@(np.linalg.pinv(B[np.arange(20)!=j])@conditional[np.arange(20)!=j])
           for j in range(20)]),False)})
    exact_edges.extend({'mode':mode,'dyad':p,'d_LM_exact':float(lm[j]),
                        'negative_d_control_sink_exact':float(sink[j]),
                        'negative_d_punctuation_exact':float(punct[j]),
                        'd_conditional_lexical_share':float(conditional[j]),
                        'paired_meanings':int(np.isfinite(E['lm_exact'][:,j]).sum())}
                       for j,p in enumerate(dyads))
exact_summary_table=pd.DataFrame(exact_rows)
display(exact_summary_table)
display(pd.DataFrame(convergence_rows))
display(pd.DataFrame(exact_edges))
exact_summary_table.to_csv(OUTPUT/'exact_tensor_sink_summary.csv',index=False)
pd.DataFrame(convergence_rows).to_csv(OUTPUT/'exact_tensor_sample_size_convergence.csv',index=False)
pd.DataFrame(exact_edges).to_csv(OUTPUT/'exact_tensor_sink_edges.csv',index=False)


# %% original code cell 6; notebook cell 13
record={'primary':PRIMARY,'cpu_only':True,'tensor_meanings_per_mode':N_TENSOR_MEANINGS,
        'tensor_sample_seed':TENSOR_SAMPLE_SEED,'tensor_sample_selection_hash':selection_hash,
        'exact_tensor_meaning_bootstrap_replicates':N_EXACT_BOOT,
        'selected_corpus_sentence_ids':selected_ids,'input_csv_sha256':input_hashes,
        'collector_run_config':config,'control_sink_definition':
        '1 minus retained source attention mass; combined removed controls, not separable by token',
        'nllb_capped_free_directions_excluded':57 if PRIMARY=='nllb' else None}
(OUTPUT/'analysis_provenance.json').write_text(
    json.dumps(record,indent=2,ensure_ascii=False,default=str),encoding='utf8')
print('Saved tables and provenance to',OUTPUT)

