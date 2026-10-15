# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1
import hashlib, json, time, unicodedata
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
N_BOOT=10000
BOOT_CHUNK=100
SEED=20260930
OUTPUT=Path('/content/drive/MyDrive/indic_40_target_fertility_cpu')/PRIMARY
OUTPUT.mkdir(parents=True,exist_ok=True)
print('CPU only:',PRIMARY,'| meanings:',EXPECTED_N[PRIMARY],
      '| bootstrap:',N_BOOT,'| output:',OUTPUT)


# %% original code cell 1; notebook cell 3
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(['1','1.0','true','yes'])

def reverse(p):
    a,b=p.split('__');return b+'__'+a

def load_teacher_forced(label):
    root=ROOTS[label]
    assert root.is_dir(),f'Missing original collector backup: {root}'
    config_path=root/'manifests'/'run_config.json'
    config=json.loads(config_path.read_text(encoding='utf8'))
    assert int(config['n_sentences'])==EXPECTED_N[label]
    assert ('nllb' if label=='nllb' else 'indictrans2') in str(config['model']).lower()
    path=root/'teacher_forced_all_stats.csv';assert path.is_file(),path
    needed={'pair','sentence_id','corpus_sentence_id','analysis_view','mean',
            'src_len','tgt_len','all_retained_mean','all_retained_src_len',
            'source_tokens_json','target_tokens_json','source_tags_removed'}
    if label=='nllb':needed.add('source_special_tokens_removed')
    df=pd.read_csv(path,usecols=lambda x:x in needed,
                   dtype={'corpus_sentence_id':str})
    assert needed.issubset(df.columns),sorted(needed-set(df.columns))
    assert len(df)==40*EXPECTED_N[label] and df.pair.nunique()==40
    assert not df.duplicated(['pair','sentence_id']).any()
    assert not df.duplicated(['pair','corpus_sentence_id']).any()
    assert df.groupby('pair').sentence_id.apply(
        lambda x:set(x.astype(int))==set(range(EXPECTED_N[label]))).all()
    assert (df.source_tags_removed.astype(int)==(1 if label=='nllb' else 2)).all()
    if label=='nllb':
        assert (df.source_special_tokens_removed.astype(int)==2).all()
    eligible=df.analysis_view.eq('lexical_no_punctuation')
    assert int(eligible.sum())==len(df),(label,'teacher-forced ineligible rows',int((~eligible).sum()))
    df['LM']=pd.to_numeric(df['mean'],errors='coerce')*pd.to_numeric(df.src_len,errors='coerce')
    df['sink_proxy']=1.0-pd.to_numeric(df.all_retained_mean,errors='coerce')*\
                          pd.to_numeric(df.all_retained_src_len,errors='coerce')
    assert df[['LM','sink_proxy','src_len','tgt_len']].notna().all().all()
    assert df.sink_proxy.between(-.002,1.002).all()
    manifest_path=root/'manifests'/'sample_identity.csv'
    manifest=pd.read_csv(manifest_path,dtype={'corpus_sentence_id':str})
    assert not manifest.duplicated(['pair','corpus_sentence_id']).any()
    assert {'source_sha256','target_sha256'}.issubset(manifest.columns)
    pairs=sorted(df.pair.unique())
    dyads=[p for p in pairs if p<reverse(p)]
    assert len(dyads)==20 and all(reverse(p) in pairs for p in dyads)
    langs=sorted({x for p in pairs for x in p.split('__')})
    assert len(langs)==11
    identity=manifest.set_index(['pair','corpus_sentence_id'])
    for p in dyads:
        a=identity.loc[p].sort_index();b=identity.loc[reverse(p)].sort_index()
        assert a.index.equals(b.index)
        assert (a.source_sha256.values==b.target_sha256.values).all()
        assert (a.target_sha256.values==b.source_sha256.values).all()
    ids=sorted(df.corpus_sentence_id.unique(),key=lambda s:(len(s),s))
    assert len(ids)==EXPECTED_N[label]
    return {'label':label,'root':root,'config':config,'path':path,
            'input_hashes':{'config':sha256_file(config_path),
                            'teacher_forced_csv':sha256_file(path),
                            'sample_identity':sha256_file(manifest_path)},
            'frame':df,'pairs':pairs,'dyads':dyads,'langs':langs,'ids':ids}

needed_labels=tuple(dict.fromkeys((PRIMARY,'pilot','holdout')))
DATA={label:load_teacher_forced(label) for label in needed_labels}
assert DATA['pilot']['langs']==DATA['holdout']['langs']
assert DATA['pilot']['dyads']==DATA['holdout']['dyads']
assert set(DATA['pilot']['ids']).isdisjoint(DATA['holdout']['ids'])
if PRIMARY=='nllb':
    assert DATA['nllb']['langs']==DATA['holdout']['langs']
    assert DATA['nllb']['dyads']==DATA['holdout']['dyads']
    assert DATA['nllb']['ids']==DATA['holdout']['ids']
    free_path=ROOTS['nllb']/'free_generation_all_stats.csv'
    free=pd.read_csv(free_path,usecols=['generation_hit_max_length',
                 'attention_inference_eligible','attention_tensor_role','analysis_view'])
    capped=truthy(free.generation_hit_max_length)
    free_eligible=free.analysis_view.eq('lexical_no_punctuation') & ~capped &\
      truthy(free.attention_inference_eligible) &\
      free.attention_tensor_role.astype(str).str.lower().eq('inferential')
    assert len(free)==30000 and int(capped.sum())==57 and int(free_eligible.sum())==29943
    print('NLLB free-generation capped rows audited: 57; excluded from this teacher-forced analysis')
print('Loaded saved teacher-forced rows:',{x:len(v['frame']) for x,v in DATA.items()})


# %% original code cell 2; notebook cell 5
def punctuation_piece(piece):
    text=str(piece).replace('▁','').replace('Ġ','').strip()
    return not text or all(unicodedata.category(ch)[0] in {'P','S','Z','C'} for ch in text)

def piece_counts(text,expected_len):
    tokens=json.loads(text)
    lexical=[p for p in tokens if not punctuation_piece(p)]
    assert len(lexical)==int(expected_len),(len(lexical),expected_len)
    starts=sum(str(p).startswith(('▁','Ġ')) for p in lexical)
    if lexical and not str(lexical[0]).startswith(('▁','Ġ')):starts+=1
    assert len(lexical)>0 and starts>0
    return len(lexical),starts

def add_token_counts(bundle):
    df=bundle['frame'].copy()
    df[['source_language','target_language']]=df.pair.str.split('__',expand=True)
    src=np.array([piece_counts(t,n) for t,n in zip(df.source_tokens_json,df.src_len)],int)
    tgt=np.array([piece_counts(t,n) for t,n in zip(df.target_tokens_json,df.tgt_len)],int)
    df['source_pieces']=src[:,0];df['source_words']=src[:,1]
    df['target_pieces']=tgt[:,0];df['target_words']=tgt[:,1]
    df['target_row_fertility']=df.target_pieces/df.target_words
    df=df.sort_values(['pair','sentence_id']).reset_index(drop=True)
    source_disagreements=int((df.groupby(['source_language','corpus_sentence_id'])
        .source_tokens_json.nunique()>1).sum())
    target_disagreements=int((df.groupby(['target_language','corpus_sentence_id'])
        .target_tokens_json.nunique()>1).sum())
    source=df.drop_duplicates(['source_language','corpus_sentence_id'])[
        ['source_language','corpus_sentence_id','source_pieces','source_words',
         'source_tokens_json']].rename(columns={'source_language':'language'})
    target=df.drop_duplicates(['target_language','corpus_sentence_id'])[
        ['target_language','corpus_sentence_id','target_pieces','target_words',
         'target_tokens_json']].rename(columns={'target_language':'language'})
    expected=len(bundle['ids'])*len(bundle['langs'])
    assert len(source)==len(target)==expected
    assert not source.duplicated(['language','corpus_sentence_id']).any()
    assert not target.duplicated(['language','corpus_sentence_id']).any()
    print(bundle['label'],'source/target tokenization disagreements across partners:',
          source_disagreements,target_disagreements)
    bundle.update(frame=df,source_unique=source,target_unique=target,
                  disagreement_counts=(source_disagreements,target_disagreements))

for bundle in DATA.values():add_token_counts(bundle)

def count_matrices(bundle,role):
    df=bundle[role+'_unique'].set_index(['corpus_sentence_id','language'])
    index=pd.MultiIndex.from_product([bundle['ids'],bundle['langs']],
                                     names=['corpus_sentence_id','language'])
    aligned=df.reindex(index)
    cols=[role+'_pieces',role+'_words']
    assert aligned[cols].notna().all().all()
    n=len(bundle['ids']);m=len(bundle['langs'])
    return (aligned[cols[0]].to_numpy(float).reshape(n,m),
            aligned[cols[1]].to_numpy(float).reshape(n,m))

def fertility_table(bundle):
    sp,sw=count_matrices(bundle,'source')
    tp,tw=count_matrices(bundle,'target')
    f_src=sp.sum(axis=0)/sw.sum(axis=0)
    f_tgt=tp.sum(axis=0)/tw.sum(axis=0)
    table=pd.DataFrame({'dataset':bundle['label'],'language':bundle['langs'],
        'f_src':f_src,'f_tgt':f_tgt,
        'source_lexical_pieces':sp.sum(axis=0).astype(int),
        'source_estimated_words':sw.sum(axis=0).astype(int),
        'target_lexical_pieces':tp.sum(axis=0).astype(int),
        'target_estimated_words':tw.sum(axis=0).astype(int)})
    bundle.update(source_piece_matrix=sp,source_word_matrix=sw,
                  target_piece_matrix=tp,target_word_matrix=tw,
                  fertility_table=table)
    return table

fertility_tables=pd.concat([fertility_table(DATA[x]) for x in needed_labels],ignore_index=True)
display(fertility_tables)
fertility_tables.to_csv(OUTPUT/'language_fertility.csv',index=False)


# %% original code cell 3; notebook cell 7
def rankdata_average(x):
    x=np.asarray(x,float)
    vals,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    assert len(vals)>1
    ends=np.cumsum(counts)
    ranks=ends-(counts-1)/2
    return ranks[inverse]

def spearman(x,y):
    a=rankdata_average(x);b=rankdata_average(y)
    return float(np.corrcoef(a,b)[0,1])

def additive_design(bundle):
    pairs,langs=bundle['pairs'],bundle['langs'];n=len(langs)
    X=np.zeros((len(pairs),1+2*n));X[:,0]=1
    for i,p in enumerate(pairs):
        a,b=p.split('__')
        X[i,1:1+n]=-1/n;X[i,1+langs.index(a)]+=1
        X[i,1+n:]=-1/n;X[i,1+n+langs.index(b)]+=1
    assert np.linalg.matrix_rank(X)==1+2*(n-1)
    return X

def directed_matrix(bundle):
    df=bundle['frame']
    pivot=df.pivot(index='corpus_sentence_id',columns='pair',values='LM')
    pivot=pivot.reindex(index=bundle['ids'],columns=bundle['pairs'])
    assert pivot.notna().all().all()
    return pivot.to_numpy(float)

bundle=DATA[PRIMARY]
X=additive_design(bundle);Xpinv=np.linalg.pinv(X)
Y=directed_matrix(bundle)
coefs=Xpinv@Y.mean(axis=0)
n=len(bundle['langs'])
alpha=coefs[1:1+n];beta=coefs[1+n:]
theta=alpha-beta
assert abs(alpha.sum())<1e-10 and abs(beta.sum())<1e-10
B=np.zeros((len(bundle['dyads']),n))
for j,p in enumerate(bundle['dyads']):
    a,b=p.split('__');B[j,bundle['langs'].index(a)]=1;B[j,bundle['langs'].index(b)]=-1
directed_means=dict(zip(bundle['pairs'],Y.mean(axis=0)))
edge=np.array([directed_means[p]-directed_means[reverse(p)] for p in bundle['dyads']])
theta_from_field=np.linalg.pinv(B)@edge
assert np.max(np.abs(theta-theta_from_field))<1e-9
observed=bundle['fertility_table'].copy()
observed['alpha_source']=alpha;observed['beta_target']=beta;observed['theta']=theta
observed['theta_from_reciprocal_field']=theta_from_field
observed['directed_mean_fitted_R2_centered']=1-\
  np.sum((Y.mean(axis=0)-X@coefs)**2)/np.sum((Y.mean(axis=0)-Y.mean())**2)
display(observed)
observed.to_csv(OUTPUT/'fertility_additive_coefficients.csv',index=False)

association_targets={'beta_target':beta,'alpha_source':alpha,'theta':theta}
f_tgt=observed.f_tgt.to_numpy(float)
rho_obs={name:spearman(f_tgt,values) for name,values in association_targets.items()}
reps=np.empty((N_BOOT,3),float)
rng=np.random.default_rng(SEED+{'pilot':1,'holdout':2,'nllb':3}[PRIMARY])
pieces=bundle['target_piece_matrix'];words=bundle['target_word_matrix']
started=time.perf_counter()
for start in range(0,N_BOOT,BOOT_CHUNK):
    count=min(BOOT_CHUNK,N_BOOT-start)
    ii=rng.integers(0,len(bundle['ids']),size=(count,len(bundle['ids'])))
    y=Y[ii].mean(axis=1)
    boot_coefs=y@Xpinv.T
    boot_f=pieces[ii].sum(axis=1)/words[ii].sum(axis=1)
    for q in range(count):
        a=boot_coefs[q,1:1+n];b=boot_coefs[q,1+n:]
        reps[start+q]=[spearman(boot_f[q],b),spearman(boot_f[q],a),
                       spearman(boot_f[q],a-b)]
    if (start+count)%1000==0:
        print('meaning bootstrap',PRIMARY,start+count,'/',N_BOOT,
              'elapsed_min',round((time.perf_counter()-started)/60,1),flush=True)

association_table=pd.DataFrame([
    {'dataset':PRIMARY,'fertility':'f_tgt','coefficient':name,'n_languages':11,
     'spearman_rho':rho_obs[name],
     'bootstrap_CI_low':float(np.quantile(reps[:,j],.025)),
     'bootstrap_CI_high':float(np.quantile(reps[:,j],.975)),
     'bootstrap_replicates':N_BOOT,'interpretation':'effect size only; no significance claim'}
    for j,name in enumerate(('beta_target','alpha_source','theta'))])
display(association_table)
association_table.to_csv(OUTPUT/'target_fertility_rank_associations.csv',index=False)


# %% original code cell 4; notebook cell 9
primary_f=DATA[PRIMARY]['fertility_table']
fs=primary_f.f_src.to_numpy(float);ft=primary_f.f_tgt.to_numpy(float)
delta_rho=abs(spearman(ft,beta))-abs(spearman(fs,beta))
sanity={'dataset':PRIMARY,'rho_fsrc_ftgt':spearman(fs,ft),
        'mean_abs_fertility_difference':float(np.mean(np.abs(fs-ft))),
        'median_abs_fertility_difference':float(np.median(np.abs(fs-ft))),
        'delta_abs_rho_ftgt_beta_minus_fsrc_beta':float(delta_rho)}
print('T3:',sanity)
pd.DataFrame([sanity]).to_csv(OUTPUT/'source_target_fertility_sanity.csv',index=False)

pilot=DATA['pilot']['fertility_table'].set_index('language').loc[bundle['langs'],'f_tgt']
holdout=DATA['holdout']['fertility_table'].set_index('language').loc[bundle['langs'],'f_tgt']
stability={'rho_pilot_holdout_target_fertility':spearman(pilot,holdout),
           'largest_abs_per_language_difference':float(np.max(np.abs(pilot-holdout))),
           'language_at_largest_difference':str((pilot-holdout).abs().idxmax())}
print('T4:',stability)
pd.DataFrame([stability]).to_csv(OUTPUT/'pilot_holdout_fertility_stability.csv',index=False)

src=DATA[PRIMARY]['source_unique'].set_index(['language','corpus_sentence_id'])
tgt=DATA[PRIMARY]['target_unique'].set_index(['language','corpus_sentence_id'])
roles=src[['source_pieces','source_words','source_tokens_json']].join(
      tgt[['target_pieces','target_words','target_tokens_json']],how='inner')
roles['piece_count_diff']=roles.target_pieces-roles.source_pieces
roles['word_count_diff']=roles.target_words-roles.source_words
roles['token_string_match']=roles.source_tokens_json.eq(roles.target_tokens_json)
role_audit=roles.groupby(level=0).agg(
    matched_token_strings=('token_string_match','sum'),
    mean_piece_count_diff=('piece_count_diff','mean'),
    max_abs_piece_count_diff=('piece_count_diff',lambda x:np.abs(x).max()),
    mean_word_count_diff=('word_count_diff','mean'),
    max_abs_word_count_diff=('word_count_diff',lambda x:np.abs(x).max()))
role_audit['n_meanings']=len(DATA[PRIMARY]['ids'])
role_audit['dataset']=PRIMARY
role_audit.reset_index().to_csv(OUTPUT/'source_target_role_tokenization_audit.csv',index=False)
if PRIMARY=='nllb' and not np.allclose(fs,ft,atol=1e-12,rtol=0):
    print('NLLB source and target fertility differ. Inspect the role-tokenization audit below;')
    print('the collector uses source encoding and text_target label encoding paths.')
    display(role_audit)
    mismatch=roles.loc[(roles.piece_count_diff!=0)|(roles.word_count_diff!=0)]
    display(mismatch.head(20))


# %% original code cell 5; notebook cell 11
def ols_cluster_meaning(frame,outcome,by_target=None):
    d=frame if by_target is None else frame.loc[frame.target_language.eq(by_target)]
    d=d.loc[np.isfinite(d[outcome])&np.isfinite(d.target_row_fertility)&
            (d.src_len>0)&(d.tgt_len>0)].copy()
    assert len(d)>0
    columns=[np.ones(len(d)),d.target_row_fertility.to_numpy(float),
             np.log1p(d.src_len.to_numpy(float)),np.log1p(d.tgt_len.to_numpy(float))]
    names=['intercept','target_pieces_per_estimated_word',
           'log1p_source_lexical_length','log1p_target_lexical_length']
    for variable in ['source_language','target_language']:
        levels=sorted(d[variable].unique())
        if variable=='target_language' and by_target is not None:continue
        for level in levels[1:]:
            columns.append(d[variable].eq(level).to_numpy(float))
            names.append(variable+'='+level)
    X=np.column_stack(columns);y=d[outcome].to_numpy(float)
    rank=np.linalg.matrix_rank(X)
    assert rank==X.shape[1],(outcome,by_target,'design rank',rank,X.shape[1])
    beta_hat=np.linalg.lstsq(X,y,rcond=None)[0]
    residual=y-X@beta_hat
    inv=np.linalg.inv(X.T@X)
    cluster_codes,cluster_ids=pd.factorize(d.corpus_sentence_id,sort=True)
    G=len(cluster_ids);N=len(d);p=X.shape[1]
    assert G>p and N>p
    meat=np.zeros((p,p))
    for g in range(G):
        z=X[cluster_codes==g].T@residual[cluster_codes==g]
        meat+=np.outer(z,z)
    correction=(G/(G-1))*((N-1)/(N-p))
    V=correction*inv@meat@inv
    se=float(np.sqrt(max(V[1,1],0)))
    slope=float(beta_hat[1])
    return {'outcome':outcome,'target_language':by_target or 'POOLED',
            'slope_per_one_extra_piece_per_word':slope,
            'cluster_SE':se,'CI95_low':slope-1.96*se,'CI95_high':slope+1.96*se,
            'n_rows':N,'n_meaning_clusters':G,'design_rank':rank,
            'n_predictors':p,'source_FE':True,'target_FE':by_target is None,
            'predictor_SD':float(d.target_row_fertility.std(ddof=1)),
            'predictor_distinct_values':int(d.target_row_fertility.nunique()),
            'controls':'log1p source and target lexical lengths'}

regression_rows=[]
for outcome in ('LM','sink_proxy'):
    regression_rows.append(ols_cluster_meaning(bundle['frame'],outcome))
    for language in bundle['langs']:
        regression_rows.append(ols_cluster_meaning(bundle['frame'],outcome,language))
regression_table=pd.DataFrame(regression_rows)
display(regression_table)
regression_table.to_csv(OUTPUT/'meaning_clustered_fertility_slopes.csv',index=False)
for outcome in ('LM','sink_proxy'):
    sub=regression_table.loc[(regression_table.outcome==outcome)&
                             (regression_table.target_language!='POOLED')]
    print(outcome,'pooled slope:',regression_table.loc[
          (regression_table.outcome==outcome)&
          (regression_table.target_language=='POOLED'),
          'slope_per_one_extra_piece_per_word'].iloc[0],
          '| negative target-language slopes:',int((sub.slope_per_one_extra_piece_per_word<0).sum()),
          'of',len(sub))


# %% original code cell 6; notebook cell 13
record={'primary':PRIMARY,'cpu_only':True,'teacher_forced_only':True,
    'n_bootstrap':N_BOOT,'bootstrap_unit':'meaning, resampled jointly across all directions and language fertility counts',
    'fertility_definition':'total lexical pieces divided by total boundary-estimated words over one record per language and meaning; punctuation rule matches previous source statistic',
    'regression':'OLS with log1p lexical lengths, source and target fixed effects; CR1 standard errors clustered by meaning',
    'sink_proxy':'1 - all_retained_mean * all_retained_src_len; all retained target-row support differs from LM',
    'Q_status':'omitted because complete same-support all-row Q requires tensors or complete head caches',
    'nllb_free_generation_capped_rows_excluded':57,
    'inputs':{label:{'root':str(x['root']),'hashes':x['input_hashes']}
              for label,x in DATA.items()}}
(OUTPUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('Saved all result tables and provenance to',OUTPUT)

