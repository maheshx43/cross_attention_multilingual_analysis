# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1
import hashlib, json, math, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from IPython.display import display

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass  # For a local CPU Python environment with the same backups mounted.

PRIMARY = 'pilot'
ROOTS = {
    'pilot': Path('/content/drive/MyDrive/indic_40_backups/indic_40_balanced_pilot_250_protocol_v3'),
    'holdout': Path('/content/drive/MyDrive/indic_40_backups/indic_40_balanced_holdout_750_rows250_999_locked_v4'),
    'nllb': Path('/content/drive/MyDrive/indic_40_backups/nllb_40_confirmatory_holdout_750_final_v6'),
}
EXPECTED_N = {'pilot':250, 'holdout':750, 'nllb':750}
EXPECTED_PAIRED_SUPPORT = {
    'pilot':{'teacher_forced':5000,'free_generation':4997},
    'holdout':{'teacher_forced':15000,'free_generation':14997},
    'nllb':{'teacher_forced':15000,'free_generation':14944},
}
OUTPUT = Path('/content/drive/MyDrive/indic_40_conservative_field_followup_stats') / PRIMARY
N_BOOT = 10000
SEED = 20260928
MODES = ('teacher_forced','free_generation')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('CPU-only primary:', PRIMARY, '| bootstrap replicates:', N_BOOT, '| output:', OUTPUT)


# %% original code cell 1; notebook cell 3
def safe_mean(x, axis=0):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    count = ok.sum(axis=axis)
    total = np.where(ok, x, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, float), where=count>0)

def reverse(pair):
    a,b = pair.split('__')
    return b+'__'+a

def truthy(col):
    return col.astype(str).str.lower().str.strip().isin(['1','1.0','true','yes'])

def load_dataset_bundle(label, modes=('teacher_forced',), detailed=False):
    root = ROOTS[label]
    assert root.is_dir(), f'Missing original collector backup: {root}'
    cfg = json.loads((root/'manifests'/'run_config.json').read_text(encoding='utf8'))
    assert int(cfg['n_sentences']) == EXPECTED_N[label]
    expected_model = 'nllb' if label=='nllb' else 'indictrans2'
    assert expected_model in str(cfg['model']).lower()
    manifest = pd.read_csv(root/'manifests'/'sample_identity.csv',
                           dtype={'corpus_sentence_id':str})
    assert not manifest.duplicated(['pair','corpus_sentence_id']).any()
    frames = {}
    for mode in modes:
        path = root/f'{mode}_all_stats.csv'
        assert path.is_file(), path
        required = {'pair','sentence_id','corpus_sentence_id','mean','src_len',
                    'tgt_len','analysis_view'}
        extra = {'source_tokens_json','target_tokens_json','source_unknown_tokens',
                 'target_unknown_tokens'} if detailed else set()
        if label=='nllb' and mode=='free_generation':
            extra |= {'generation_hit_max_length','attention_inference_eligible',
                      'attention_tensor_role'}
        df = pd.read_csv(path, usecols=lambda c: c in required|extra,
                         dtype={'corpus_sentence_id':str})
        assert required.issubset(df.columns)
        assert len(df)==40*EXPECTED_N[label] and df.pair.nunique()==40
        assert not df.duplicated(['pair','corpus_sentence_id']).any()
        assert not df.duplicated(['pair','sentence_id']).any()
        assert df.groupby('pair').sentence_id.apply(
            lambda s: set(s.astype(int))==set(range(EXPECTED_N[label]))).all()
        assert set(df['analysis_view'].dropna().unique()).issubset(
            {'lexical_no_punctuation','lexical_no_punctuation_unavailable',
             'excluded_generation_cap'})
        eligible = df.analysis_view.eq('lexical_no_punctuation')
        if label=='nllb' and mode=='free_generation':
            assert {'generation_hit_max_length','attention_inference_eligible',
                    'attention_tensor_role'}.issubset(df.columns)
            capped = truthy(df.generation_hit_max_length)
            assert int(capped.sum())==57, 'NLLB capped-row count differs from collector'
            eligible &= ~capped & truthy(df.attention_inference_eligible)
            eligible &= df.attention_tensor_role.astype(str).str.lower().eq('inferential')
            assert int(eligible.sum())==29943, 'NLLB eligible-row count differs from collector'
        df['eligible'] = eligible
        df['lm'] = np.where(eligible,
            pd.to_numeric(df['mean'],errors='coerce')*
            pd.to_numeric(df['src_len'],errors='coerce'), np.nan)
        assert df.loc[eligible,'lm'].notna().all()
        frames[mode] = df
    pairs = sorted(frames['teacher_forced'].pair.unique())
    assert all(sorted(df.pair.unique())==pairs for df in frames.values())
    dyads = [p for p in pairs if p<reverse(p)]
    assert len(dyads)==20 and all(reverse(p) in pairs for p in dyads)
    langs = sorted({x for p in pairs for x in p.split('__')})
    assert len(langs)==11
    B = np.zeros((20,11))
    for j,p in enumerate(dyads):
        a,b=p.split('__'); B[j,langs.index(a)]=1; B[j,langs.index(b)]=-1
    assert np.linalg.matrix_rank(B)==10
    ids=sorted(frames['teacher_forced'].corpus_sentence_id.unique(),key=lambda s:(len(s),s))
    assert len(ids)==EXPECTED_N[label]
    identity=manifest.set_index(['pair','corpus_sentence_id'])
    for p in dyads:
        q=reverse(p)
        a=identity.loc[p].sort_index(); b=identity.loc[q].sort_index()
        assert a.index.equals(b.index)
        assert (a.source_sha256.values==b.target_sha256.values).all()
        assert (a.target_sha256.values==b.source_sha256.values).all()
    edges={}; directed={}
    for mode,df in frames.items():
        assert all(set(g.corpus_sentence_id)==set(ids) for _,g in df.groupby('pair'))
        pivot=df.pivot(index='corpus_sentence_id',columns='pair',values='lm').reindex(index=ids,columns=pairs)
        # Match the two directions within each meaning before averaging. This
        # matters for NLLB free generation, where capped rows are ineligible.
        matched_directed={}
        for p in dyads:
            q=reverse(p)
            complete=np.isfinite(pivot[p])&np.isfinite(pivot[q])
            matched_directed[p]=float(pivot.loc[complete,p].mean())
            matched_directed[q]=float(pivot.loc[complete,q].mean())
        directed[mode]=np.array([matched_directed[p] for p in pairs])
        edges[mode]=np.stack([(pivot[p]-pivot[reverse(p)]).to_numpy(float) for p in dyads],axis=1)
        assert np.isfinite(safe_mean(edges[mode],0)).all()
        assert int(np.isfinite(edges[mode]).sum())==EXPECTED_PAIRED_SUPPORT[label][mode]
    return dict(label=label,root=root,config=cfg,manifest=manifest,frames=frames,edges=edges,
                directed=directed,pairs=pairs,dyads=dyads,langs=langs,B=B,ids=ids)

DATA = {name:load_dataset_bundle(name, MODES,
                                 detailed=(name==PRIMARY)) for name in ROOTS}
assert DATA['pilot']['dyads']==DATA['holdout']['dyads']==DATA['nllb']['dyads']
assert DATA['pilot']['langs']==DATA['holdout']['langs']==DATA['nllb']['langs']
assert set(DATA['pilot']['ids']).isdisjoint(DATA['holdout']['ids'])
assert DATA['holdout']['ids']==DATA['nllb']['ids']
for col in ['source_sha256','target_sha256']:
    left=DATA['holdout']['manifest'].set_index(['pair','corpus_sentence_id'])[col].sort_index()
    right=DATA['nllb']['manifest'].set_index(['pair','corpus_sentence_id'])[col].sort_index()
    assert left.equals(right), f'IndicTrans2/NLLB holdout text mismatch: {col}'
print({name:{m:int(np.isfinite(e).sum()) for m,e in b['edges'].items()} for name,b in DATA.items()})


# %% original code cell 2; notebook cell 5
def field_fit(bundle, edge):
    B=bundle['B']; y=safe_mean(edge,0)
    assert np.isfinite(y).all()
    theta=np.linalg.pinv(B)@y
    g=B@theta; r=y-g
    energy=float(np.dot(g,g)/np.dot(y,y))
    return dict(edge=y,theta=theta,gradient=g,residual=r,
                energy=energy,residual_norm=float(np.linalg.norm(r)))

FITS={mode:field_fit(DATA[PRIMARY],E) for mode,E in DATA[PRIMARY]['edges'].items()}
display(pd.DataFrame([{'mode':m,'n_meanings':len(DATA[PRIMARY]['ids']),
                       'edge_energy_fraction':f['energy'],
                       'residual_l2':f['residual_norm'],
                       'max_abs_residual':np.abs(f['residual']).max()}
                      for m,f in FITS.items()]))


# %% original code cell 3; notebook cell 7
def rank_stats(a,b):
    return float(kendalltau(a,b).statistic),float(spearmanr(a,b).statistic)

def compare_with_meaning_bootstrap(left,right,mode='teacher_forced',n_boot=N_BOOT):
    A,C=DATA[left],DATA[right]
    assert A['dyads']==C['dyads'] and A['langs']==C['langs']
    E1,E2=A['edges'][mode],C['edges'][mode]
    ids1,ids2=A['ids'],C['ids']
    paired=ids1==ids2
    if paired:
        assert E1.shape[0]==E2.shape[0]
    P=np.linalg.pinv(A['B'])
    t1=P@safe_mean(E1,0); t2=P@safe_mean(E2,0)
    observed=rank_stats(t1,t2)
    rng=np.random.default_rng(SEED+({'pilot':1,'holdout':2,'nllb':3}[left])*10+
                              {'pilot':1,'holdout':2,'nllb':3}[right])
    reps=np.empty((n_boot,2))
    for k in range(n_boot):
        idx1=rng.integers(0,len(ids1),len(ids1))
        idx2=idx1 if paired else rng.integers(0,len(ids2),len(ids2))
        v1=P@safe_mean(E1[idx1],0);v2=P@safe_mean(E2[idx2],0)
        reps[k]=rank_stats(v1,v2)
    return {'comparison':left+' vs '+right,'paired_meaning_resampling':paired,
            'kendall_tau':observed[0],
            'kendall_boot_lo':np.quantile(reps[:,0],.025),
            'kendall_boot_hi':np.quantile(reps[:,0],.975),
            'spearman_rho':observed[1],
            'spearman_boot_lo':np.quantile(reps[:,1],.025),
            'spearman_boot_hi':np.quantile(reps[:,1],.975),
            'bootstrap_replicates':n_boot}

comparison_table=pd.DataFrame([
    {'mode':mode,**compare_with_meaning_bootstrap(left,right,mode)}
    for mode in MODES
    for left,right in [('pilot','holdout'),('holdout','nllb')]
])
display(comparison_table)
comparison_table.to_csv(OUTPUT/'theta_rank_comparison.csv',index=False)


# %% original code cell 4; notebook cell 9
def punctuation_piece(piece):
    text=str(piece).replace('▁','').replace('Ġ','').strip()
    return not text or all(unicodedata.category(ch)[0] in {'P','S','Z','C'} for ch in text)

def source_covariates(bundle):
    df=bundle['frames']['teacher_forced'].copy()
    assert 'source_tokens_json' in df.columns
    df['language']=df.pair.str.split('__').str[0]
    df=df.sort_values('pair')
    groups=df.groupby(['language','corpus_sentence_id'],sort=False)
    # Source tokenization should not depend on which target was paired with it.
    disagreements=sum(g.source_tokens_json.nunique()>1 for _,g in groups)
    unique=df.drop_duplicates(['language','corpus_sentence_id'])
    rows=[]
    for row in unique.itertuples(index=False):
        tokens=json.loads(row.source_tokens_json)
        lexical=[p for p in tokens if not punctuation_piece(p)]
        assert len(lexical)==int(row.src_len), (row.pair,row.corpus_sentence_id)
        starts=sum(str(p).startswith(('▁','Ġ')) for p in lexical)
        # The first segment may lack an explicit boundary marker.
        if lexical and not str(lexical[0]).startswith(('▁','Ġ')): starts+=1
        chars=[len(str(p).lstrip('▁Ġ')) for p in lexical if str(p)!='<unk>']
        unk=sum(str(p)=='<unk>' for p in lexical)
        rows.append({'language':row.language,'lexical_pieces':len(lexical),
                     'estimated_words':starts,'subword_chars':sum(chars),
                     'non_unk_pieces':len(chars),'unk_pieces':unk})
    counts=pd.DataFrame(rows).groupby('language').sum(numeric_only=True)
    table=counts.reset_index()[['language']].copy()
    table['script']=table.language.str.split('_').str[1]
    table['fertility_pieces_per_estimated_word']=(counts.lexical_pieces/counts.estimated_words).values
    table['mean_subword_length_chars']=(counts.subword_chars/counts.non_unk_pieces).values
    table['source_unk_rate']=(counts.unk_pieces/counts.lexical_pieces).values
    table['log_training_data_size']=np.nan
    table['theta_teacher_forced']=[FITS['teacher_forced']['theta'][bundle['langs'].index(x)]
                                   for x in table.language]
    print('Source tokenization disagreements across targets:',disagreements)
    return table

covariate_table=source_covariates(DATA[PRIMARY])
display(covariate_table)
covariate_table.to_csv(OUTPUT/'language_covariates.csv',index=False)

association_rows=[]
predictors=['fertility_pieces_per_estimated_word','mean_subword_length_chars',
            'source_unk_rate','log_training_data_size']
for name in predictors:
    x=covariate_table[name].to_numpy(float)
    y=covariate_table.theta_teacher_forced.to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()>=5 and len(np.unique(x[ok]))>=3:
        z=spearmanr(x[ok],y[ok])
        association_rows.append({'predictor':name,'n':int(ok.sum()),
                                 'spearman_rho':float(z.statistic),
                                 'p_exploratory':float(z.pvalue),'status':'computed'})
    else:
        association_rows.append({'predictor':name,'n':int(ok.sum()),
                                 'spearman_rho':np.nan,'p_exploratory':np.nan,
                                 'status':'unavailable or insufficient variation'})
for script,n in covariate_table.script.value_counts().items():
    if 2<=n<=9:
        x=(covariate_table.script==script).astype(int)
        z=spearmanr(x,covariate_table.theta_teacher_forced)
        association_rows.append({'predictor':'script_'+script,'n':11,
                                 'spearman_rho':float(z.statistic),
                                 'p_exploratory':float(z.pvalue),'status':'computed'})
association_table=pd.DataFrame(association_rows)
display(association_table)
association_table.to_csv(OUTPUT/'theta_covariate_rank_associations.csv',index=False)


# %% original code cell 5; notebook cell 11
def residual_bootstrap(bundle,mode,n_boot=N_BOOT):
    E=bundle['edges'][mode];B=bundle['B'];P=np.linalg.pinv(B)
    fit=field_fit(bundle,E);n=len(E)
    # Recenter each edge's finite observations so its empirical mean is exactly B theta.
    # NaNs and within-meaning cross-edge dependence remain in their original positions.
    E0=np.where(np.isfinite(E),E-fit['edge']+fit['gradient'],np.nan)
    assert np.allclose(safe_mean(E0,0),fit['gradient'],atol=1e-12)
    rng=np.random.default_rng(SEED+100+('teacher_forced'==mode)+
                              {'pilot':1,'holdout':2,'nllb':3}[bundle['label']]*10)
    boot_r=np.empty((n_boot,20));boot_norm=np.empty(n_boot)
    null_norm=np.empty(n_boot);boot_energy=np.empty(n_boot)
    for k in range(n_boot):
        ii=rng.integers(0,n,n)
        y=safe_mean(E[ii],0); y0=safe_mean(E0[ii],0)
        assert np.isfinite(y).all() and np.isfinite(y0).all()
        g=B@(P@y);r=y-g
        r0=y0-B@(P@y0)
        boot_r[k]=r;boot_norm[k]=np.linalg.norm(r)
        null_norm[k]=np.linalg.norm(r0)
        boot_energy[k]=np.dot(g,g)/np.dot(y,y)
    obs=fit['residual_norm']
    summary={'mode':mode,'observed_residual_l2':obs,
             'bootstrap_residual_l2_lo':np.quantile(boot_norm,.025),
             'bootstrap_residual_l2_hi':np.quantile(boot_norm,.975),
             'observed_gradient_energy':fit['energy'],
             'gradient_energy_boot_lo':np.quantile(boot_energy,.025),
             'gradient_energy_boot_hi':np.quantile(boot_energy,.975),
             'exact_field_noise_null_q95':np.quantile(null_norm,.95),
             'exact_field_noise_null_q99':np.quantile(null_norm,.99),
             'exact_field_noise_p':(1+np.sum(null_norm>=obs))/(n_boot+1),
             'bootstrap_replicates':n_boot}
    per_edge=pd.DataFrame({'dyad':bundle['dyads'],'mode':mode,
                           'observed_residual':fit['residual'],
                           'bootstrap_lo':np.quantile(boot_r,.025,axis=0),
                           'bootstrap_hi':np.quantile(boot_r,.975,axis=0)})
    return summary,per_edge

residual_summaries=[];residual_edges=[]
for mode in MODES:
    a,b=residual_bootstrap(DATA[PRIMARY],mode)
    residual_summaries.append(a);residual_edges.append(b)
residual_summary=pd.DataFrame(residual_summaries)
residual_edge_table=pd.concat(residual_edges,ignore_index=True)
display(residual_summary)
display(residual_edge_table)
residual_summary.to_csv(OUTPUT/'residual_bootstrap_summary.csv',index=False)
residual_edge_table.to_csv(OUTPUT/'residual_bootstrap_edges.csv',index=False)


# %% original code cell 6; notebook cell 13
def additive_check(bundle,mode):
    langs=bundle['langs'];pairs=bundle['pairs'];n=len(langs)
    y=bundle['directed'][mode]
    assert np.isfinite(y).all()
    X=np.zeros((len(pairs),1+2*n));X[:,0]=1
    for i,p in enumerate(pairs):
        a,b=p.split('__')
        X[i,1:1+n]=-1/n;X[i,1+langs.index(a)]+=1
        X[i,1+n:]=-1/n;X[i,1+n+langs.index(b)]+=1
    coef=np.linalg.pinv(X)@y
    mu=coef[0];alpha=coef[1:1+n];beta=coef[1+n:]
    fitted=X@coef
    theta_add=alpha-beta
    theta_graph=field_fit(bundle,bundle['edges'][mode])['theta']
    # On a reciprocal graph this identity follows algebraically from the design.
    discrepancy=float(np.max(np.abs(theta_add-theta_graph)))
    assert discrepancy<1e-9, discrepancy
    sse=np.sum((y-fitted)**2)
    dyad_fit=np.array([fitted[pairs.index(p)]-fitted[pairs.index(reverse(p))]
                       for p in bundle['dyads']])
    dyad_obs=field_fit(bundle,bundle['edges'][mode])['edge']
    return ({'mode':mode,'directed_R2_centered':1-sse/np.sum((y-y.mean())**2),
             'directed_R2_against_zero':1-sse/np.sum(y*y),
             'reciprocal_R2_against_zero':1-np.sum((dyad_obs-dyad_fit)**2)/np.sum(dyad_obs**2),
             'max_abs_theta_vs_alpha_minus_beta':discrepancy,
             'n_directions':len(pairs),'design_rank':int(np.linalg.matrix_rank(X))},
            pd.DataFrame({'mode':mode,'language':langs,'alpha_source':alpha,
                          'beta_target':beta,'alpha_minus_beta':theta_add,
                          'theta_from_reciprocal_field':theta_graph}))

additive_summaries=[];additive_potentials=[]
for mode in MODES:
    a,b=additive_check(DATA[PRIMARY],mode)
    additive_summaries.append(a);additive_potentials.append(b)
additive_summary=pd.DataFrame(additive_summaries)
display(additive_summary)
display(pd.concat(additive_potentials,ignore_index=True))
additive_summary.to_csv(OUTPUT/'directed_additive_summary.csv',index=False)
pd.concat(additive_potentials,ignore_index=True).to_csv(OUTPUT/'directed_additive_potentials.csv',index=False)


# %% original code cell 7; notebook cell 15
def nllb_unknown_audit(bundle,mode):
    df=bundle['frames'][mode].copy()
    assert {'source_tokens_json','target_tokens_json','source_unknown_tokens',
            'target_unknown_tokens'}.issubset(df.columns)
    result=df[['pair','sentence_id','corpus_sentence_id','eligible','lm',
               'source_unknown_tokens','target_unknown_tokens']].copy()
    result['lm_without_unk']=result.lm
    result['source_unk_attention_mass']=np.where(result.eligible,0.0,np.nan)
    result['reconstructed_lm_from_saved_tensor']=np.nan
    todo=df.index[df.eligible & ((df.source_unknown_tokens>0)|
                                 (df.target_unknown_tokens>0))]
    print(mode,'eligible rows containing <unk>:',len(todo),
          '| capped/ineligible rows skipped:',int((~df.eligible).sum()))
    for k,i in enumerate(todo):
        row=df.loc[i]
        source=json.loads(row.source_tokens_json);target=json.loads(row.target_tokens_json)
        src_lex=np.array([j for j,p in enumerate(source) if not punctuation_piece(p)],int)
        tgt_lex=np.array([j for j,p in enumerate(target) if not punctuation_piece(p)],int)
        src_keep=np.array([j for j in src_lex if source[j]!='<unk>'],int)
        tgt_keep=np.array([j for j in tgt_lex if target[j]!='<unk>'],int)
        src_unk=np.array([j for j in src_lex if source[j]=='<unk>'],int)
        assert len(src_lex)==int(row.src_len) and len(tgt_lex)==int(row.tgt_len)
        path=bundle['root']/mode/row.pair/f'{int(row.sentence_id)}.npy'
        assert path.is_file(), f'Missing original eligible LHTS tensor: {path}'
        full=np.load(path,mmap_mode='r',allow_pickle=False)
        assert full.ndim==4 and full.shape[-2:]==(len(target),len(source))
        matrix=np.asarray(full,dtype=np.float32).mean(axis=(0,1))
        old=matrix[np.ix_(tgt_lex,src_lex)].sum(axis=1).mean()
        new=(matrix[np.ix_(tgt_keep,src_keep)].sum(axis=1).mean()
             if len(tgt_keep) and len(src_keep) else np.nan)
        unk_mass=(matrix[np.ix_(tgt_lex,src_unk)].sum(axis=1).mean()
                  if len(src_unk) else 0.0)
        result.at[i,'reconstructed_lm_from_saved_tensor']=old
        result.at[i,'lm_without_unk']=new
        result.at[i,'source_unk_attention_mass']=unk_mass
        if (k+1)%100==0: print('Processed',k+1,'/',len(todo))
    delta=np.abs(result.loc[todo,'reconstructed_lm_from_saved_tensor']-
                 result.loc[todo,'lm'])
    print('Maximum saved-tensor vs CSV lexical-mass difference:',
          float(delta.max()) if len(delta) else 0.0)
    assert not len(delta) or float(delta.max())<0.005, 'Tensor reconstruction mismatch'
    # Missing/ineligible observations remain NaN. Never fill capped rows with zero.
    assert result.loc[~result.eligible,'lm_without_unk'].isna().all()
    # Excluding <unk> may leave no lexical source or target token in a row.
    # Such rows become missing in the sensitivity analysis, never zero.
    newly_empty=int((result.eligible & result.lm_without_unk.isna()).sum())
    print('Rows newly empty after excluding <unk>:',newly_empty)
    result['source_language']=result.pair.str.split('__').str[0]
    result['target_language']=result.pair.str.split('__').str[1]
    result.to_csv(OUTPUT/f'nllb_{mode}_unk_row_audit.csv',index=False)
    by_source=result.loc[result.eligible].groupby('source_language').agg(
        eligible_directions=('lm','size'),
        source_unk_tokens=('source_unknown_tokens','sum'),
        mean_source_unk_attention_mass=('source_unk_attention_mass','mean'))
    total_source_tokens=df.loc[df.eligible].groupby(
        df.loc[df.eligible,'pair'].str.split('__').str[0])['src_len'].sum()
    by_source['source_unk_rate']=by_source.source_unk_tokens/total_source_tokens
    by_source=by_source.reset_index().rename(columns={'source_language':'language'})
    by_target=result.loc[result.eligible].groupby('target_language').agg(
        target_unk_tokens=('target_unknown_tokens','sum'))
    total_target_tokens=df.loc[df.eligible].groupby(
        df.loc[df.eligible,'pair'].str.split('__').str[1])['tgt_len'].sum()
    by_target['target_unk_rate']=by_target.target_unk_tokens/total_target_tokens
    by_source=by_source.merge(by_target.reset_index().rename(
        columns={'target_language':'language'}),on='language',how='left')
    by_source['mode']=mode
    # Rebuild the paired field using the same meaning IDs and the same eligibility mask.
    pivot=result.pivot(index='corpus_sentence_id',columns='pair',
                       values='lm_without_unk').reindex(index=bundle['ids'],columns=bundle['pairs'])
    E=np.stack([(pivot[p]-pivot[reverse(p)]).to_numpy(float)
                for p in bundle['dyads']],axis=1)
    before=field_fit(bundle,bundle['edges'][mode]);after=field_fit(bundle,E)
    assert np.all(~np.isfinite(E) | np.isfinite(bundle['edges'][mode]))
    tau,rho=rank_stats(before['theta'],after['theta'])
    compare={'mode':mode,'eligible_direction_rows':int(result.eligible.sum()),
             'paired_meaning_edge_observations':int(np.isfinite(E).sum()),
             'rows_recomputed_from_tensors':len(todo),
             'newly_empty_direction_rows':newly_empty,
             'gradient_energy_original':before['energy'],
             'gradient_energy_excluding_unk':after['energy'],
             'gradient_energy_change':after['energy']-before['energy'],
             'max_abs_theta_change':float(np.max(np.abs(after['theta']-before['theta']))),
             'theta_kendall_tau':tau,'theta_spearman_rho':rho}
    return by_source,compare,pd.DataFrame({'mode':mode,'language':bundle['langs'],
                                           'theta_original':before['theta'],
                                           'theta_excluding_unk':after['theta']})

if PRIMARY=='nllb':
    unk_languages=[];unk_comparisons=[];unk_potentials=[]
    for mode in MODES:
        a,b,c=nllb_unknown_audit(DATA[PRIMARY],mode)
        unk_languages.append(a);unk_comparisons.append(b);unk_potentials.append(c)
    display(pd.concat(unk_languages,ignore_index=True))
    display(pd.DataFrame(unk_comparisons))
    display(pd.concat(unk_potentials,ignore_index=True))
    pd.concat(unk_languages,ignore_index=True).to_csv(OUTPUT/'nllb_unk_by_language.csv',index=False)
    pd.DataFrame(unk_comparisons).to_csv(OUTPUT/'nllb_unk_field_comparison.csv',index=False)
    pd.concat(unk_potentials,ignore_index=True).to_csv(OUTPUT/'nllb_unk_theta_comparison.csv',index=False)
else:
    print('NLLB <unk> audit is run in the NLLB sibling notebook.')


# %% original code cell 8; notebook cell 17
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

record={'primary':PRIMARY,'cpu_only':True,'seed':SEED,
        'bootstrap_replicates':N_BOOT,
        'teacher_forced_primary':True,
        'resource_proxy':'unavailable in permitted saved artifacts',
        'fertility_definition':'saved lexical pieces per SentencePiece boundary-estimated word',
        'nllb_capped_free_generation_directions_excluded':57,
        'inputs':{name:{'root':str(bundle['root']),
                        'run_config':bundle['config'],
                        'csv_sha256':{mode:sha256_file(bundle['root']/f'{mode}_all_stats.csv')
                                      for mode in MODES}}
                  for name,bundle in DATA.items()}}
(OUTPUT/'analysis_provenance.json').write_text(
    json.dumps(record,indent=2,ensure_ascii=False,default=str),encoding='utf8')
print('Saved analysis tables and provenance to',OUTPUT)

