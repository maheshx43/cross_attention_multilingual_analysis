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

PRIMARY='holdout'
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
CACHE_ROOT=Path('/content/drive/MyDrive/indic_40_head_layer_svd_cpu')/PRIMARY
OUTPUT=CACHE_ROOT/'rankk_followup'
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
    cache_dir=CACHE_ROOT/'head_cache'/f'{mode}_{input_hashes[mode][:12]}'
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
RANKS=(1,2,3,5,10)
N_PRED_NULL=5000
N_SPLIT=1000
SEED_FOLLOWUP=20260930
METRICS=('LM','Q','combined_sink')
ADJUSTMENTS=('raw','length_adjusted')
CENTERINGS=('uncentered','head_centered')
H=len(matrices['teacher_forced']['LM']['raw'])
LAYER_N,HEAD_N=expected_shape
assert H==LAYER_N*HEAD_N and HEAD_N>=2
assert np.linalg.matrix_rank(B)==10

def graph_preserving_null(M,rng):
    # Independently relabel each head's potential; preserve its two energies.
    theta=M@np.linalg.pinv(B).T
    gradient=theta@B.T
    residual=M-gradient
    n=len(M)
    lang_order=np.argsort(rng.random((n,len(langs))),axis=1)
    theta_perm=np.take_along_axis(theta,lang_order,axis=1)
    g=theta_perm@B.T
    old=np.linalg.norm(gradient,axis=1);new=np.linalg.norm(g,axis=1)
    g*=np.divide(old,new,out=np.zeros_like(old),where=new>1e-12)[:,None]
    edge_order=np.argsort(rng.random((n,20)),axis=1)
    r=np.take_along_axis(residual,edge_order,axis=1)@CYCLE
    old=np.linalg.norm(residual,axis=1);new=np.linalg.norm(r,axis=1)
    r*=np.divide(old,new,out=np.zeros_like(old),where=new>1e-12)[:,None]
    result=g+r
    assert np.allclose(np.sum(result**2,axis=1),np.sum(M**2,axis=1),rtol=1e-5,atol=1e-8)
    return result

MASKS=[np.arange(20)!=j for j in range(20)]
POTENTIAL_MAPS=[np.linalg.pinv(B[mask]).T for mask in MASKS]
assert all(np.linalg.matrix_rank(B[mask])==10 for mask in MASKS)

def right_basis(X,kmax=10):
    gram=X.T@X
    eigval,V=np.linalg.eigh(gram)
    order=np.argsort(eigval)[::-1]
    return V[:,order[:kmax]]

def predict_all_heldout(M,centering):
    # H × edges × ranks predictions; no held-out edge enters its fit.
    preds=np.empty((len(M),20,len(RANKS)),float)
    for j,mask in enumerate(MASKS):
        theta=M[:,mask]@POTENTIAL_MAPS[j]
        mu=theta.mean(axis=0,keepdims=True) if centering=='head_centered' else np.zeros((1,11))
        X=theta-mu
        V=right_basis(X)
        for ri,k in enumerate(RANKS):
            th=mu+(X@V[:,:k])@V[:,:k].T
            preds[:,j,ri]=th@B[j]
    assert np.isfinite(preds).all()
    return preds

def heldout_r2(M,preds):
    den=float(np.sum(M*M))
    assert den>0
    return 1-np.sum((M[:,:,None]-preds)**2,axis=(0,1))/den

def gini(values):
    a=np.sort(np.maximum(np.asarray(values,float),0))
    if a.sum()<=0:return np.nan
    n=len(a)
    return float((2*np.arange(1,n+1)-n-1)@a/(n*a.sum()))

def concentration_rows(values,kind,labels):
    a=np.maximum(np.asarray(values,float),0)
    total=a.sum();sorted_a=np.sort(a)[::-1]
    record=dict(labels,signal=kind,total_signal=float(total),gini=gini(a),
                heads_with_positive_signal=int((a>0).sum()))
    for pct in (5,10,20):
        n=max(1,int(np.ceil(len(a)*pct/100)))
        record[f'top_{pct}pct_share']=float(sorted_a[:n].sum()/total) if total>0 else np.nan
    return record

def stratified_halves(rng):
    left=[];right=[]
    for layer in range(LAYER_N):
        indexes=rng.permutation(np.arange(layer*HEAD_N,(layer+1)*HEAD_N))
        cut=len(indexes)//2
        left.extend(indexes[:cut]);right.extend(indexes[cut:])
    return np.asarray(left,int),np.asarray(right,int)

def split_score(M,left,right,centering):
    theta=M@np.linalg.pinv(B).T
    a=theta[left];b=theta[right]
    if centering=='head_centered':
        a=a-a.mean(axis=0,keepdims=True)
        b=b-b.mean(axis=0,keepdims=True)
    Va=right_basis(a);Vb=right_basis(b)
    scores=[]
    for k in (1,3,5):
        va=Va[:,:k];vb=Vb[:,:k]
        cross_ab=float(np.sum((b@va)**2)/np.sum(b*b))
        cross_ba=float(np.sum((a@vb)**2)/np.sum(a*a))
        cosines=np.linalg.svd(va.T@vb,compute_uv=False).clip(0,1)
        angles=np.degrees(np.arccos(cosines))
        scores.append((k,(cross_ab+cross_ba)/2,float(np.mean(angles)),float(np.max(angles))))
    return scores

prediction_rows=[];dyad_rows=[];split_rows=[];concentration=[];curve_rows=[]
head_rows=[];layer_rows=[]
total_scenarios=len(MODES)*len(METRICS)*len(ADJUSTMENTS)*len(CENTERINGS)
scenario_i=0
for mode in MODES:
  for metric in METRICS:
    for adjustment in ADJUSTMENTS:
      M=matrices[mode][metric][adjustment]
      assert M.shape==(H,20) and np.isfinite(M).all()
      for centering in CENTERINGS:
        scenario_i+=1
        labels={'mode':mode,'metric':metric,'adjustment':adjustment,'centering':centering}
        start=time.perf_counter()
        pred=predict_all_heldout(M,centering)
        observed=heldout_r2(M,pred)
        null_scores=np.empty((N_PRED_NULL,len(RANKS)),float)
        rng=np.random.default_rng(SEED_FOLLOWUP+100000*scenario_i)
        for rep in range(N_PRED_NULL):
            N=graph_preserving_null(M,rng)
            null_scores[rep]=heldout_r2(N,predict_all_heldout(N,centering))
            if (rep+1)%1000==0:
                print('rank-k null',scenario_i,'/',total_scenarios,labels,
                      rep+1,'/',N_PRED_NULL,'elapsed_min',
                      round((time.perf_counter()-start)/60,1),flush=True)
        for ri,k in enumerate(RANKS):
            prediction_rows.append(dict(labels,rank=k,strict_dyad_holdout=(adjustment=='raw'),
                  observed_R2_zero=float(observed[ri]),
                  null_median_R2_zero=float(np.median(null_scores[:,ri])),
                  null_q025_R2_zero=float(np.quantile(null_scores[:,ri],.025)),
                  null_q975_R2_zero=float(np.quantile(null_scores[:,ri],.975)),
                  improvement_over_null_median=float(observed[ri]-np.median(null_scores[:,ri])),
                  p_null_ge_observed=float((1+np.sum(null_scores[:,ri]>=observed[ri]))/(N_PRED_NULL+1)),
                  n_null=N_PRED_NULL))
            for j,dyad in enumerate(dyads):
                y=M[:,j];yp=pred[:,j,ri]
                dyad_rows.append(dict(labels,rank=k,dyad=dyad,
                    strict_dyad_holdout=(adjustment=='raw'),
                    R2_zero=float(1-np.sum((y-yp)**2)/np.sum(y*y)),
                    RMSE=float(np.sqrt(np.mean((y-yp)**2))),
                    edge_RMS=float(np.sqrt(np.mean(y*y)))))

        # Split-half comparisons use the same graph-preserving null and
        # layer-stratified head assignments for each observed/null draw.
        obs_split={k:[] for k in (1,3,5)}
        null_split={k:[] for k in (1,3,5)}
        for rep in range(N_SPLIT):
            left,right=stratified_halves(rng)
            N=graph_preserving_null(M,rng)
            for k,frac,mean_angle,max_angle in split_score(M,left,right,centering):
                obs_split[k].append((frac,mean_angle,max_angle))
            for k,frac,mean_angle,max_angle in split_score(N,left,right,centering):
                null_split[k].append((frac,mean_angle,max_angle))
            if (rep+1)%250==0:
                print('split-half',scenario_i,'/',total_scenarios,
                      rep+1,'/',N_SPLIT,flush=True)
        for k in (1,3,5):
            o=np.asarray(obs_split[k]);n=np.asarray(null_split[k])
            split_rows.append(dict(labels,rank=k,n_splits=N_SPLIT,
                cross_half_explained_median=float(np.median(o[:,0])),
                cross_half_explained_q025=float(np.quantile(o[:,0],.025)),
                cross_half_explained_q975=float(np.quantile(o[:,0],.975)),
                null_cross_half_median=float(np.median(n[:,0])),
                null_cross_half_q975=float(np.quantile(n[:,0],.975)),
                fraction_null_draws_ge_observed_median=float(np.mean(n[:,0]>=np.median(o[:,0]))),
                mean_principal_angle_median_deg=float(np.median(o[:,1])),
                largest_principal_angle_median_deg=float(np.median(o[:,2])),
                null_mean_principal_angle_median_deg=float(np.median(n[:,1])),
                random_subspace_expected_fraction=k/10))

        A=M-M.mean(axis=0,keepdims=True) if centering=='head_centered' else M
        _,_,Vt=np.linalg.svd(A,full_matrices=False)
        component1=(A@Vt[0])**2
        component3=np.sum((A@Vt[:3].T)**2,axis=1)
        idx3=RANKS.index(3)
        gain=np.sum(M*M-(M-pred[:,:,idx3])**2,axis=1)
        positive_gain=np.maximum(gain,0)
        for kind,values in [('component1_energy',component1),
                            ('top3_energy',component3),
                            ('positive_rank3_LOEO_gain',positive_gain)]:
            concentration.append(concentration_rows(values,kind,labels))
            descending=np.sort(values)[::-1]
            for n in range(1,H+1):
                curve_rows.append(dict(labels,signal=kind,heads=n,
                    head_fraction=n/H,
                    cumulative_signal_share=float(descending[:n].sum()/descending.sum())
                    if descending.sum()>0 else np.nan))
        for h in range(H):
            head_rows.append(dict(labels,layer=h//HEAD_N,head=h%HEAD_N,
                component1_energy=float(component1[h]),top3_energy=float(component3[h]),
                rank3_LOEO_gain=float(gain[h]),positive_rank3_LOEO_gain=float(positive_gain[h]),
                rank3_R2_zero=float(1-np.sum((M[h]-pred[h,:,idx3])**2)/np.sum(M[h]**2))))
        for layer in range(LAYER_N):
            sl=slice(layer*HEAD_N,(layer+1)*HEAD_N)
            layer_rows.append(dict(labels,layer=layer,
                component1_energy_share=float(component1[sl].sum()/component1.sum()),
                top3_energy_share=float(component3[sl].sum()/component3.sum()),
                positive_rank3_LOEO_gain_share=float(positive_gain[sl].sum()/positive_gain.sum())
                    if positive_gain.sum()>0 else np.nan,
                rank3_R2_zero=float(1-np.sum((M[sl]-pred[sl,:,idx3])**2)/np.sum(M[sl]**2))))
        print('complete',scenario_i,'/',total_scenarios,labels,
              'rank3_R2',round(float(observed[idx3]),4),
              'scenario_min',round((time.perf_counter()-start)/60,1),flush=True)

prediction_table=pd.DataFrame(prediction_rows)
split_table=pd.DataFrame(split_rows)
concentration_table=pd.DataFrame(concentration)
dyad_table=pd.DataFrame(dyad_rows)
head_table=pd.DataFrame(head_rows)
layer_table=pd.DataFrame(layer_rows)
curve_table=pd.DataFrame(curve_rows)
for name,table in [('rankk_LOEO_vs_null',prediction_table),('rankk_LOEO_by_dyad',dyad_table),
                   ('split_half_stability',split_table),('head_concentration',concentration_table),
                   ('head_cumulative_signal',curve_table),('per_head_signal',head_table),
                   ('per_layer_signal',layer_table)]:
    table.to_csv(OUTPUT/f'{name}.csv',index=False)
display(prediction_table.query("mode=='teacher_forced' and metric=='LM' and adjustment=='raw'"))
display(split_table.query("mode=='teacher_forced' and metric=='LM' and adjustment=='raw'"))
display(concentration_table.query("mode=='teacher_forced' and metric=='LM' and adjustment=='raw'"))


# %% original code cell 5; notebook cell 11
# Primary teacher-forced lexical-mass maps. Other scenarios remain in CSVs.
primary=head_table.loc[(head_table['mode']=='teacher_forced') &
  (head_table['metric']=='LM') & (head_table['adjustment']=='raw') &
  (head_table['centering']=='uncentered')].copy()
assert len(primary)==H
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for ax,col,title in [(axes[0,0],'component1_energy','Top component energy'),
                     (axes[0,1],'positive_rank3_LOEO_gain','Positive rank-3 LOEO gain')]:
    grid=primary.pivot(index='layer',columns='head',values=col).to_numpy()
    im=ax.imshow(grid,aspect='auto',origin='lower',cmap='magma')
    ax.set_title(title);ax.set_xlabel('Head (zero-based)');ax.set_ylabel('Layer (zero-based)')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
layers=layer_table.loc[(layer_table['mode']=='teacher_forced') &
  (layer_table['metric']=='LM') & (layer_table['adjustment']=='raw') &
  (layer_table['centering']=='uncentered')]
axes[1,0].plot(layers.layer,layers.component1_energy_share,label='Top component')
axes[1,0].plot(layers.layer,layers.positive_rank3_LOEO_gain_share,label='Rank-3 gain')
axes[1,0].axhline(1/LAYER_N,color='0.6',linestyle=':',label='Uniform layer share')
axes[1,0].set(xlabel='Layer (zero-based)',ylabel='Share',title='Signal share by layer')
axes[1,0].legend()
curve=curve_table.loc[(curve_table['mode']=='teacher_forced') &
  (curve_table['metric']=='LM') & (curve_table['adjustment']=='raw') &
  (curve_table['centering']=='uncentered')]
for signal,label in [('component1_energy','Top component'),
                     ('top3_energy','Top 3 components'),
                     ('positive_rank3_LOEO_gain','Positive rank-3 gain')]:
    sub=curve.loc[curve.signal==signal]
    axes[1,1].plot(sub.head_fraction,sub.cumulative_signal_share,label=label)
axes[1,1].plot([0,1],[0,1],color='0.6',linestyle=':',label='Equal contribution')
axes[1,1].set(xlabel='Fraction of heads, highest first',ylabel='Cumulative signal share',
              title='Concentration across heads')
axes[1,1].legend()
fig.savefig(OUTPUT/'teacher_forced_LM_head_localization.png',dpi=180)
plt.show()

record={'primary':PRIMARY,'cpu_only':True,'all_eligible_sentences':True,
  'input_sha256':input_hashes,'source_root':str(root),
  'reused_cache_root':str(CACHE_ROOT/'head_cache'),
  'n_prediction_null':N_PRED_NULL,'n_split_half':N_SPLIT,'seed':SEED_FOLLOWUP,
  'ranks':list(RANKS),'layers':LAYER_N,'heads_per_layer':HEAD_N,
  'null':'Independent per-head language-potential permutation plus cycle-space residual permutation, rescaled to preserve per-head gradient and residual norms.',
  'prediction':'For each omitted dyad, fit per-head potentials and a rank-k language basis using only the other 19 dyads.',
  'centering':'Head-centered rank-k reconstruction adds back the training-head mean potential; it is not parameter matched to uncentered rank k.',
  'length_adjusted_LOEO_limit':'Length slopes are pooled across all dyads; only raw LOEO is a strict held-out edge prediction.',
  'head_concentration':'Squared edge-SVD projection by head; positive LOEO gain is clipped only for concentration summaries.',
  'notes':['NLLB capped free-generation rows stay excluded.',
           'All dyad fields use only meanings eligible in both directions.',
           'Combined sink cannot be decomposed into EOS and language-tag shares from cropped tensors.']}
(OUTPUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('Saved tables, figure and provenance to',OUTPUT)

