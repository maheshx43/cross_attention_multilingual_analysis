# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1
import hashlib, json, time, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
EXPECTED_ELIGIBLE={'pilot':{'teacher_forced':10000,'free_generation':9997},
                   'holdout':{'teacher_forced':30000,'free_generation':29997},
                   'nllb':{'teacher_forced':30000,'free_generation':29943}}
EXPECTED_PAIRED={'pilot':{'teacher_forced':5000,'free_generation':4997},
                 'holdout':{'teacher_forced':15000,'free_generation':14997},
                 'nllb':{'teacher_forced':15000,'free_generation':14944}}
MODES=('teacher_forced','free_generation')
N_NULL=5000
SEED=20260929
root=ROOTS[PRIMARY]
OUTPUT=Path('/content/drive/MyDrive/indic_40_head_layer_svd_cpu')/PRIMARY
OUTPUT.mkdir(parents=True,exist_ok=True)
print('CPU only:',PRIMARY,'| full eligible sentences | output:',OUTPUT)


# %% original code cell 1; notebook cell 3
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def reverse(pair):
    a,b=pair.split('__');return b+'__'+a

def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(['1','1.0','true','yes'])

assert root.is_dir(),f'Missing saved GPU collector backup: {root}'
config_path=root/'manifests'/'run_config.json'
config=json.loads(config_path.read_text(encoding='utf8'))
assert int(config['n_sentences'])==EXPECTED_N[PRIMARY]
assert ('nllb' if PRIMARY=='nllb' else 'indictrans2') in str(config['model']).lower()
input_hashes={'config':sha256_file(config_path)}
frames={}
for mode in MODES:
    path=root/f'{mode}_all_stats.csv';assert path.is_file(),path
    input_hashes[mode]=sha256_file(path)
    needed={'pair','sentence_id','corpus_sentence_id','analysis_view','mean',
            'src_len','tgt_len','source_tokens_json','target_tokens_json'}
    if PRIMARY=='nllb' and mode=='free_generation':
        needed|={'generation_hit_max_length','attention_inference_eligible','attention_tensor_role'}
    df=pd.read_csv(path,usecols=lambda c:c in needed,dtype={'corpus_sentence_id':str})
    assert needed.issubset(df.columns),sorted(needed-set(df.columns))
    assert len(df)==40*EXPECTED_N[PRIMARY] and df.pair.nunique()==40
    assert not df.duplicated(['pair','corpus_sentence_id']).any()
    assert not df.duplicated(['pair','sentence_id']).any()
    eligible=df.analysis_view.eq('lexical_no_punctuation')
    if PRIMARY=='nllb' and mode=='free_generation':
        capped=truthy(df.generation_hit_max_length)
        assert int(capped.sum())==57
        eligible &= ~capped & truthy(df.attention_inference_eligible)
        eligible &= df.attention_tensor_role.astype(str).str.lower().eq('inferential')
    df['eligible']=eligible
    assert int(eligible.sum())==EXPECTED_ELIGIBLE[PRIMARY][mode]
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
P=B@np.linalg.pinv(B);CYCLE=np.eye(20)-P
ids=sorted(frames['teacher_forced'].corpus_sentence_id.unique(),key=lambda s:(len(s),s))
assert len(ids)==EXPECTED_N[PRIMARY]
identity_path=root/'manifests'/'sample_identity.csv'
identity=pd.read_csv(identity_path,dtype={'corpus_sentence_id':str})
assert {'pair','corpus_sentence_id','source_sha256','target_sha256'}.issubset(identity.columns)
assert not identity.duplicated(['pair','corpus_sentence_id']).any()
input_hashes['sample_identity']=sha256_file(identity_path)
identity=identity.set_index(['pair','corpus_sentence_id'])
for p in dyads:
    a=identity.loc[p].sort_index();b=identity.loc[reverse(p)].sort_index()
    assert a.index.equals(b.index)
    assert (a.source_sha256.values==b.target_sha256.values).all()
    assert (a.target_sha256.values==b.source_sha256.values).all()
for mode,df in frames.items():
    assert df.groupby('pair').corpus_sentence_id.apply(lambda x:set(x)==set(ids)).all()
print('Eligible saved tensor rows:',{m:int(f.eligible.sum()) for m,f in frames.items()})


# %% original code cell 2; notebook cell 5
def punctuation_piece(piece):
    text=str(piece).replace('▁','').replace('Ġ','').strip()
    return not text or all(unicodedata.category(ch)[0] in {'P','S','Z','C'} for ch in text)

def extract_head_row(row,mode):
    source=json.loads(row.source_tokens_json);target=json.loads(row.target_tokens_json)
    src_lex=np.array([j for j,p in enumerate(source) if not punctuation_piece(p)],int)
    tgt_lex=np.array([j for j,p in enumerate(target) if not punctuation_piece(p)],int)
    assert len(src_lex)==int(row.src_len) and len(tgt_lex)==int(row.tgt_len)
    assert len(src_lex)>0 and len(tgt_lex)>0
    path=root/mode/row.pair/f'{int(row.sentence_id)}.npy'
    assert path.is_file(),f'Missing saved eligible LHTS tensor: {path}'
    full=np.load(path,mmap_mode='r',allow_pickle=False)
    assert full.ndim==4 and full.shape[-2:]==(len(target),len(source)),path
    # Saved source controls are cropped without renormalizing the retained rows.
    matrix=np.asarray(full,dtype=np.float32)
    lexical_rows=np.take(matrix,tgt_lex,axis=-2)
    retained=lexical_rows.sum(axis=-1).mean(axis=-1)
    lexical=np.take(lexical_rows,src_lex,axis=-1).sum(axis=-1).mean(axis=-1)
    assert np.isfinite(retained).all() and np.isfinite(lexical).all()
    assert np.all(retained>=-0.002) and np.all(retained<=1.002)
    assert np.all(lexical>=-0.002) and np.all(lexical<=retained+0.002)
    expected=float(row.mean)*int(row.src_len)
    assert abs(float(lexical.mean())-expected)<.005,(path,lexical.mean(),expected)
    return lexical.astype(np.float32),retained.astype(np.float32)

def cached_pair(mode,p,df,cache_dir):
    subset=df.loc[df.pair.eq(p)&df.eligible].sort_values('sentence_id')
    # Pandas to_numpy() would produce dtype=object, which requires pickle to reload.
    expected_ids=np.asarray(subset.corpus_sentence_id.astype(str).tolist(),dtype=np.str_)
    expected_sentence_ids=subset.sentence_id.astype(int).to_numpy()
    path=cache_dir/f'{p}.npz'
    cache_hit=path.is_file()
    needs_migration=False
    if cache_hit:
        with np.load(path,allow_pickle=False) as z:
            got_sentence_ids=z['sentence_ids'].astype(int)
            lm=z['LM'];retained=z['retained']
            try:
                got_ids=z['meaning_ids'].astype(str)
            except ValueError as exc:
                if 'Object arrays cannot be loaded' not in str(exc):raise
                # Old caches used an object array for IDs. Validate numeric rows
                # and sentence IDs below, then rewrite without unpickling anything.
                got_ids=expected_ids
                needs_migration=True
        assert np.array_equal(got_ids,expected_ids) and np.array_equal(got_sentence_ids,expected_sentence_ids)
    else:
        lm_rows=[];retained_rows=[]
        pair_started=time.perf_counter();last_report=pair_started
        for row_i,row in enumerate(subset.itertuples(index=False),1):
            l,r=extract_head_row(row,mode)
            lm_rows.append(l);retained_rows.append(r)
            now=time.perf_counter()
            if now-last_report>=30 or row_i==len(subset):
                elapsed=max(now-pair_started,1e-6)
                remaining=(len(subset)-row_i)*elapsed/row_i/60
                print(mode,p,'tensor rows',row_i,'/',len(subset),
                      'pair_elapsed_min',round(elapsed/60,1),
                      'pair_ETA_min',round(remaining,1),flush=True)
                last_report=now
        lm=np.stack(lm_rows);retained=np.stack(retained_rows)
    assert lm.shape==retained.shape and len(lm)==len(subset)
    assert lm.ndim==3 and np.isfinite(lm).all() and np.isfinite(retained).all()
    assert np.max(np.abs(lm.mean(axis=(1,2))-
                         subset['mean'].to_numpy(float)*subset.src_len.to_numpy(float)))<.005
    if not cache_hit or needs_migration:
        tmp=path.with_suffix('.tmp')
        with open(tmp,'wb') as f:
            np.savez(f,meaning_ids=expected_ids,sentence_ids=expected_sentence_ids,
                     LM=lm,retained=retained,
                     src_len=subset.src_len.to_numpy(int),tgt_len=subset.tgt_len.to_numpy(int))
        tmp.replace(path)
        if needs_migration:
            print(mode,p,'migrated existing object-ID cache without tensor reread',flush=True)
    return {'ids':expected_ids,'LM':lm,'retained':retained,
            'src_len':subset.src_len.to_numpy(float),
            'tgt_len':subset.tgt_len.to_numpy(float),'cache_hit':cache_hit}

head_caches={};extraction_rows=[];expected_shape=None
for mode in MODES:
    df=frames[mode]
    cache_dir=OUTPUT/'head_cache'/f'{mode}_{input_hashes[mode][:12]}'
    cache_dir.mkdir(parents=True,exist_ok=True)
    saved={};started=time.perf_counter()
    for pair_i,p in enumerate(pairs,1):
        saved[p]=cached_pair(mode,p,df,cache_dir)
        shape=saved[p]['LM'].shape[1:]
        if expected_shape is None:expected_shape=shape
        assert shape==expected_shape,(mode,p,shape,expected_shape)
        elapsed=(time.perf_counter()-started)/60
        print(mode,'pair',pair_i,'/40','rows',len(saved[p]['ids']),
              'cache_hit',saved[p]['cache_hit'],
              'mode_elapsed_min',round(elapsed,1),flush=True)
    head_caches[mode]=saved
    extraction_rows.append({'mode':mode,'eligible_rows':sum(len(x['ids']) for x in saved.values()),
                            'layers':expected_shape[0],'heads':expected_shape[1],
                            'elapsed_minutes':(time.perf_counter()-started)/60,
                            'cache_directory':str(cache_dir)})
assert extraction_rows[0]['eligible_rows']==EXPECTED_ELIGIBLE[PRIMARY]['teacher_forced']
assert extraction_rows[1]['eligible_rows']==EXPECTED_ELIGIBLE[PRIMARY]['free_generation']
display(pd.DataFrame(extraction_rows))
pd.DataFrame(extraction_rows).to_csv(OUTPUT/'extraction_support.csv',index=False)


# %% original code cell 3; notebook cell 7
def paired_data(cache):
    paired=[];counts=[]
    for dyad in dyads:
        a=cache[dyad];b=cache[reverse(dyad)]
        shared=np.intersect1d(a['ids'],b['ids'])
        assert len(shared)>0
        ia=pd.Index(a['ids']).get_indexer(shared)
        ib=pd.Index(b['ids']).get_indexer(shared)
        assert (ia>=0).all() and (ib>=0).all()
        lm_a=a['LM'][ia].reshape(len(shared),-1);lm_b=b['LM'][ib].reshape(len(shared),-1)
        ret_a=a['retained'][ia].reshape(len(shared),-1)
        ret_b=b['retained'][ib].reshape(len(shared),-1)
        q_a=np.divide(lm_a,ret_a,out=np.full_like(lm_a,np.nan),where=ret_a>1e-8)
        q_b=np.divide(lm_b,ret_b,out=np.full_like(lm_b,np.nan),where=ret_b>1e-8)
        X=np.column_stack([
            np.log1p(a['src_len'][ia])-np.log1p(b['src_len'][ib]),
            np.log1p(a['tgt_len'][ia])-np.log1p(b['tgt_len'][ib])])
        paired.append({'dyad':dyad,'ids':shared,'length_delta':X,
                       'LM':lm_a-lm_b,
                       'Q':q_a-q_b,
                       'combined_sink':ret_b-ret_a})
        counts.append(len(shared))
    return paired,np.array(counts,int)

def matrix_and_length_adjustment(paired,metric):
    Y=[e[metric] for e in paired]
    X=[e['length_delta'] for e in paired]
    nheads=Y[0].shape[1]
    means=np.column_stack([np.nanmean(v,axis=0) for v in Y])
    support=np.column_stack([np.isfinite(v).sum(axis=0) for v in Y])
    all_y=np.concatenate(Y,axis=0);all_x=np.concatenate(X,axis=0)
    adjusted=np.empty_like(means)
    slopes=np.empty((nheads,2))
    for h in range(nheads):
        valid=np.isfinite(all_y[:,h])
        assert valid.sum()>=int(.6*len(all_y)),(metric,h,int(valid.sum()))
        beta=np.linalg.lstsq(all_x[valid],all_y[valid,h],rcond=None)[0]
        slopes[h]=beta
        for j,(x,y) in enumerate(zip(X,Y)):
            mask=np.isfinite(y[:,h])
            assert mask.sum()>=max(20,int(.6*len(y))),(metric,h,j,int(mask.sum()))
            adjusted[h,j]=means[h,j]-x[mask].mean(axis=0)@beta
    assert np.isfinite(means).all() and np.isfinite(adjusted).all()
    return means,adjusted,support,slopes

matrices={};support_rows=[];slope_rows=[]
for mode in MODES:
    paired,counts=paired_data(head_caches[mode])
    assert int(counts.sum())==EXPECTED_PAIRED[PRIMARY][mode],(mode,counts.tolist())
    matrices[mode]={}
    for metric in ['LM','Q','combined_sink']:
        raw,adjusted,support,slopes=matrix_and_length_adjustment(paired,metric)
        matrices[mode][metric]={'raw':raw,'length_adjusted':adjusted,
                                'support':support,'length_slopes':slopes}
        support_rows.append({'mode':mode,'metric':metric,
                             'paired_meaning_edge_rows':int(counts.sum()),
                             'min_paired_meanings_per_dyad':int(counts.min()),
                             'min_head_edge_finite_rows':int(support.min()),
                             'undefined_head_direction_values':int((np.concatenate(
                                 [e[metric] for e in paired],axis=0).size-support.sum()))})
        for h in range(raw.shape[0]):
            slope_rows.append({'mode':mode,'metric':metric,'layer':h//expected_shape[1],
                               'head':h%expected_shape[1],
                               'slope_delta_log_source_lexical_length':float(slopes[h,0]),
                               'slope_delta_log_target_lexical_length':float(slopes[h,1])})
    # The head average must reproduce the published full-data lexical field.
    csv=frames[mode].copy()
    csv['LM_csv']=np.where(csv.eligible,csv['mean']*csv.src_len,np.nan)
    csv_lookup=csv.set_index(['pair','corpus_sentence_id'])['LM_csv']
    expected=[]
    for e in paired:
        f=csv_lookup.loc[[(e['dyad'],s) for s in e['ids']]].to_numpy(float)
        r=csv_lookup.loc[[(reverse(e['dyad']),s) for s in e['ids']]].to_numpy(float)
        expected.append(float(np.mean(f-r)))
    difference=np.max(np.abs(matrices[mode]['LM']['raw'].mean(axis=0)-expected))
    assert difference<.005,(mode,difference)
    print(mode,'paired per dyad:',counts.tolist(),
          '| max tensor vs CSV edge difference:',float(difference))

support_table=pd.DataFrame(support_rows);display(support_table)
support_table.to_csv(OUTPUT/'field_support.csv',index=False)
pd.DataFrame(slope_rows).to_csv(OUTPUT/'length_slopes.csv',index=False)


# %% original code cell 4; notebook cell 9
def spectrum(A):
    s=np.linalg.svd(A,compute_uv=False)
    energy=s*s
    total=energy.sum()
    assert total>0
    proportions=energy/total
    entropy=-np.sum(proportions[proportions>0]*np.log(proportions[proportions>0]))
    return {'top1':float(proportions[:1].sum()),
            'top3':float(proportions[:3].sum()),
            'top5':float(proportions[:5].sum()),
            'effective_rank':float(np.exp(entropy)),
            'singular_values':s,'proportions':proportions}

def graph_preserving_null(M,rng):
    theta=M@np.linalg.pinv(B).T
    gradient=theta@B.T
    residual=M-gradient
    n=len(M)
    lang_order=np.argsort(rng.random((n,len(langs))),axis=1)
    theta_perm=np.take_along_axis(theta,lang_order,axis=1)
    g=theta_perm@B.T
    gnorm=np.linalg.norm(gradient,axis=1)
    gn=np.linalg.norm(g,axis=1)
    g*=np.divide(gnorm,gn,out=np.zeros_like(gnorm),where=gn>1e-12)[:,None]
    edge_order=np.argsort(rng.random((n,20)),axis=1)
    r=np.take_along_axis(residual,edge_order,axis=1)@CYCLE
    rnorm=np.linalg.norm(residual,axis=1)
    rn=np.linalg.norm(r,axis=1)
    r*=np.divide(rnorm,rn,out=np.zeros_like(rnorm),where=rn>1e-12)[:,None]
    return g+r

loeo_weights=[]
for j in range(20):
    mask=np.arange(20)!=j
    assert np.linalg.matrix_rank(B[mask])==10
    loeo_weights.append(B[j]@np.linalg.pinv(B[mask]))

svd_rows=[];head_rows=[];curve_rows=[]
for mode in MODES:
    for metric in ['LM','Q','combined_sink']:
        for adjustment in ['raw','length_adjusted']:
            M=matrices[mode][metric][adjustment]
            gradient=M@P.T
            loeo=np.column_stack([M[:,np.arange(20)!=j]@loeo_weights[j]
                                  for j in range(20)])
            ge=np.divide(np.sum(gradient**2,axis=1),np.sum(M**2,axis=1),
                         out=np.full(len(M),np.nan),where=np.sum(M**2,axis=1)>0)
            loeo_r2=np.divide(np.sum((M-loeo)**2,axis=1),np.sum(M**2,axis=1),
                              out=np.full(len(M),np.nan),where=np.sum(M**2,axis=1)>0)
            for h in range(len(M)):
                head_rows.append({'mode':mode,'metric':metric,'adjustment':adjustment,
                                  'layer':h//expected_shape[1],
                                  'head':h%expected_shape[1],
                                  'gradient_energy':float(ge[h]),
                                  'LOEO_R2_zero':float(1-loeo_r2[h]),
                                  'edge_RMS':float(np.sqrt(np.mean(M[h]**2)))})
            for centering in ['uncentered','head_centered']:
                A=M if centering=='uncentered' else M-M.mean(axis=0,keepdims=True)
                obs=spectrum(A)
                row={'mode':mode,'metric':metric,'adjustment':adjustment,
                     'centering':centering,'n_heads':len(M),
                     'aggregate_gradient_energy':float(
                         np.sum((M.mean(axis=0)@P.T)**2)/
                         np.sum(M.mean(axis=0)**2)),
                     'median_head_gradient_energy':float(np.nanmedian(ge)),
                     'median_head_LOEO_R2_zero':float(np.nanmedian(1-loeo_r2)),
                     'top1':obs['top1'],'top3':obs['top3'],'top5':obs['top5'],
                     'effective_rank':obs['effective_rank']}
                for k,(s,p) in enumerate(zip(obs['singular_values'],obs['proportions']),1):
                    curve_rows.append({'mode':mode,'metric':metric,'adjustment':adjustment,
                                       'centering':centering,'component':k,
                                       'singular_value':float(s),'energy_fraction':float(p),
                                       'cumulative_energy':float(np.sum(obs['proportions'][:k]))})
                if metric in ['LM','Q']:
                    rng=np.random.default_rng(SEED+1000*(mode=='free_generation')+
                          100*(metric=='Q')+10*(adjustment=='length_adjusted'))
                    null=np.empty((N_NULL,3))
                    for rep in range(N_NULL):
                        N=graph_preserving_null(M,rng)
                        if centering=='head_centered':N=N-N.mean(axis=0,keepdims=True)
                        sp=spectrum(N)
                        null[rep]=sp['top1'],sp['top3'],sp['top5']
                        if (rep+1)%1000==0:
                            print('null progress:',mode,metric,adjustment,centering,
                                  rep+1,'/',N_NULL,flush=True)
                    for col,k in [('top1',0),('top3',1),('top5',2)]:
                        row[f'{col}_null_median']=float(np.median(null[:,k]))
                        row[f'{col}_null_q95']=float(np.quantile(null[:,k],.95))
                        row[f'{col}_null_p_concentrated']=float(
                            (1+np.sum(null[:,k]>=obs[col]))/(N_NULL+1))
                svd_rows.append(row)
                print(mode,metric,adjustment,centering,
                      'top1',round(obs['top1'],4),'top3',round(obs['top3'],4),flush=True)

svd_table=pd.DataFrame(svd_rows);head_table=pd.DataFrame(head_rows)
display(svd_table)
display(head_table.groupby(['mode','metric','adjustment'])[
    ['gradient_energy','LOEO_R2_zero','edge_RMS']].median())
svd_table.to_csv(OUTPUT/'svd_vs_graph_null.csv',index=False)
head_table.to_csv(OUTPUT/'per_head_conservativeness.csv',index=False)
pd.DataFrame(curve_rows).to_csv(OUTPUT/'singular_spectra.csv',index=False)


# %% original code cell 5; notebook cell 11
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for row,mode in enumerate(MODES):
    for col,metric in enumerate(['LM','Q']):
        ax=axes[row,col]
        for adjustment,style in [('raw','-'),('length_adjusted','--')]:
            sub=pd.DataFrame(curve_rows)
            sub=sub.loc[(sub['mode']==mode)&(sub['metric']==metric)&
                        (sub['adjustment']==adjustment)&
                        (sub['centering']=='head_centered')]
            ax.plot(sub.component,sub.cumulative_energy,style,
                    label=adjustment)
        ax.axvline(10,color='0.65',linewidth=1)
        ax.set_title(f'{mode}: {metric}, head-centered')
        ax.set_xlim(1,20);ax.set_ylim(0,1.02)
        ax.set_xlabel('SVD component');ax.set_ylabel('Cumulative edge energy')
        ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT/'head_layer_scree.png',dpi=180)
plt.show()

record={'primary':PRIMARY,'cpu_only':True,'all_eligible_sentences':True,
        'source_root':str(root),'input_sha256':input_hashes,
        'n_null':N_NULL,'seed':SEED,
        'tensor_shape_layers_heads':list(expected_shape),
        'definitions':{'LM':'per-head lexical-source mass averaged over lexical target rows',
                       'Q':'per-head LM divided by retained-source mass on same lexical target rows',
                       'combined_sink':'1 minus retained-source mass; removed controls not separable',
                       'length_adjustment':'through-origin regression of meaning-level reciprocal differences on differences in log1p lexical source and target lengths'},
        'null':'independent per-head language-potential permutation and cycle-space residual permutation; gradient/residual norms preserved',
        'notes':['Each dyad averages only meanings eligible in both directions.',
                 'NLLB capped free-generation directions remain missing.',
                 'Low rank does not imply adapter factorization.']}
(OUTPUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('Saved all tables, plot, and provenance to',OUTPUT)

