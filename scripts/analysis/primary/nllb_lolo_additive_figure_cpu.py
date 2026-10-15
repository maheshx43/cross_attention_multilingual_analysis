# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 1

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass
PRIMARY='nllb'
assert PRIMARY in ('pilot','holdout','nllb')
MODES=('teacher_forced',)
OLD_PILOT=Path('/content/drive/MyDrive/indic_40_backups/indic_40_balanced_pilot_250_protocol_v3')
FP32_ROOT=Path('/content/drive/MyDrive/fp32_tag_intervention')
OUTPUT=Path('/content/drive/MyDrive/indic_40_lolo_mainfigure_cpu')/PRIMARY
OUTPUT.mkdir(parents=True,exist_ok=True)
EXPECTED_N={'pilot':250,'holdout':450,'nllb':450}
EXPECTED_SAMPLE_HASH='64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c'
print('CPU-only teacher-forced analysis:',PRIMARY,'meanings:',EXPECTED_N[PRIMARY])


# %% original code cell 1; notebook cell 3

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def reverse(p):
    a,b=p.split('__');return b+'__'+a
def safe_mean(a,axis=0):
    a=np.asarray(a,float);good=np.isfinite(a)
    n=good.sum(axis=axis);total=np.where(good,a,0.).sum(axis=axis)
    return np.divide(total,n,out=np.full(np.shape(total),np.nan,float),where=n>0)

input_hashes={};run_settings={}
if PRIMARY=='pilot':
    root=OLD_PILOT
    stats=root/'teacher_forced_all_stats.csv'
    manifest_path=root/'manifests'/'sample_identity.csv'
    config_path=root/'manifests'/'run_config.json'
    assert stats.is_file() and manifest_path.is_file() and config_path.is_file()
    run_settings=json.loads(config_path.read_text(encoding='utf-8'))
    assert int(run_settings['n_sentences'])==250
    assert 'indictrans2' in str(run_settings['model']).lower()
    raw=pd.read_csv(stats,dtype={'corpus_sentence_id':str})
    assert len(raw)==40*250 and raw.pair.nunique()==40
    assert raw.analysis_view.eq('lexical_no_punctuation').all()
    raw['lm']=pd.to_numeric(raw['mean'],errors='raise')*pd.to_numeric(raw['src_len'],errors='raise')
    df=raw[['pair','sentence_id','corpus_sentence_id','lm']].copy()
    input_hashes[str(stats)]=sha256_file(stats)
else:
    root=FP32_ROOT/('holdout' if PRIMARY=='holdout' else 'confirm')
    folder=root/'baseline_tf'
    manifest_path=FP32_ROOT/'sample_identity.csv'
    config_path=root/'run_lock.json'
    split_path=FP32_ROOT/'frozen_meaning_split.json'
    assert folder.is_dir() and manifest_path.is_file() and config_path.is_file() and split_path.is_file()
    run_settings=json.loads(config_path.read_text(encoding='utf-8'))
    split=json.loads(split_path.read_text(encoding='utf-8'))
    assert split['sample_hash']==EXPECTED_SAMPLE_HASH
    assert len(split['baseline450'])==450 and len(split['intervention300'])==300
    assert set(split['baseline450']).isdisjoint(split['intervention300'])
    assert run_settings['split_hash']==EXPECTED_SAMPLE_HASH
    assert ('nllb' if PRIMARY=='nllb' else 'indictrans2') in run_settings['model'].lower()
    files=sorted(folder.glob('*.csv'))
    assert len(files)==40,('Expected exactly 40 fp32 baseline CSVs',folder,len(files))
    chunks=[]
    for path in files:
        one=pd.read_csv(path,dtype={'corpus_sentence_id':str})
        need={'sentence_id','corpus_sentence_id','source','tag','text','eligible','lm'}
        assert need<=set(one.columns),(path,need-set(one.columns))
        good=one.eligible.astype(str).str.strip().str.lower().isin(('true','1','1.0'))
        assert len(one)==450 and good.all(),(path,len(one),int(good.sum()))
        assert one.tag.eq(one.text).all(),path
        assert set(one.sentence_id.astype(int))==set(split['baseline450']),path
        pair=one.source.astype(str)+'__'+one.tag.astype(str)
        assert pair.nunique()==1 and path.name==str(pair.iloc[0])+'__'+str(one.tag.iloc[0])+'.csv',path
        one['pair']=pair
        chunks.append(one[['pair','sentence_id','corpus_sentence_id','lm']])
        input_hashes[str(path)]=sha256_file(path)
    df=pd.concat(chunks,ignore_index=True)
    assert df.pair.nunique()==40 and len(df)==40*450
    input_hashes[str(split_path)]=sha256_file(split_path)

manifest=pd.read_csv(manifest_path,dtype={'corpus_sentence_id':str})
input_hashes[str(manifest_path)]=sha256_file(manifest_path)
input_hashes[str(config_path)]=sha256_file(config_path)
assert not manifest.duplicated(['pair','corpus_sentence_id']).any()
assert not manifest.duplicated(['pair','sentence_id']).any()
assert not df.duplicated(['pair','sentence_id']).any()
assert not df.duplicated(['pair','corpus_sentence_id']).any()
assert np.isfinite(pd.to_numeric(df.lm,errors='coerce')).all()
lookup=manifest.set_index(['pair','sentence_id'])
for row in df.itertuples(index=False):
    assert str(lookup.loc[(row.pair,int(row.sentence_id)),'corpus_sentence_id'])==str(row.corpus_sentence_id)
frames={'teacher_forced':df}
pairs=sorted(df.pair.unique())
dyads=[p for p in pairs if p<reverse(p)]
assert len(pairs)==40 and len(dyads)==20 and all(reverse(p) in pairs for p in dyads)
langs=sorted({l for p in pairs for l in p.split('__')})
assert len(langs)==11
B=np.zeros((20,11))
for j,p in enumerate(dyads):
    a,b=p.split('__');B[j,langs.index(a)]=1;B[j,langs.index(b)]=-1
assert np.linalg.matrix_rank(B)==10
ids=sorted(df.corpus_sentence_id.unique(),key=lambda s:(len(s),s))
assert len(ids)==EXPECTED_N[PRIMARY]
assert df.groupby('pair').corpus_sentence_id.nunique().eq(EXPECTED_N[PRIMARY]).all()
pv=df.pivot(index='corpus_sentence_id',columns='pair',values='lm').reindex(index=ids,columns=pairs)
assert np.isfinite(pv.to_numpy(float)).all()
matrices={'teacher_forced':{'lm':pv}}
edges={'teacher_forced':{'lm':np.stack([(pv[p]-pv[reverse(p)]).to_numpy(float)
                                       for p in dyads],axis=1)}}
directed={'teacher_forced':{'lm':np.array([float(pv[p].mean()) for p in pairs])}}
assert int(np.isfinite(edges['teacher_forced']['lm']).sum())==20*EXPECTED_N[PRIMARY]
print('Saved input root:',root,'| directed rows:',len(df),'| complete reciprocal meaning-edges:',20*EXPECTED_N[PRIMARY])


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

def ten_language_design(direction_names,language_names):
    n=len(language_names)
    X=np.ones((len(direction_names),1+2*n),float)
    X[:,1:]=0.
    for i,p in enumerate(direction_names):
        a,b=p.split('__')
        X[i,1+language_names.index(a)]=1
        X[i,1+n+language_names.index(b)]=1
    return X

def refit_coefficients(direction_names,y,language_names):
    # Centered source and target contrasts fix the two gauge freedoms.
    n=len(language_names)
    X=ten_language_design(direction_names,language_names)
    Z=np.column_stack([np.ones(len(direction_names)),
        X[:,1:1+n]-X[:,1:1+n].mean(axis=1,keepdims=True),
        X[:,1+n:]-X[:,1+n:].mean(axis=1,keepdims=True)])
    rank=int(np.linalg.matrix_rank(Z))
    expected=1+2*(n-1)
    if rank!=expected:
        return None,{'rank':rank,'expected_rank':expected,'identifiable':False}
    c=np.linalg.pinv(Z)@np.asarray(y,float)
    alpha=c[1:1+n];beta=c[1+n:1+2*n]
    alpha-=alpha.mean();beta-=beta.mean()
    return {'mu':float(c[0]),'alpha':alpha,'beta':beta,'theta':alpha-beta,
            'fitted':Z@c}, {'rank':rank,'expected_rank':expected,'identifiable':True}

def rank_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    rx=pd.Series(x).rank().to_numpy(float)
    ry=pd.Series(y).rank().to_numpy(float)
    return float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 and np.std(ry)>0 else np.nan

full_rows=[];deletion_rows=[];coefficient_rows=[]
for mode in MODES:
    y=directed[mode]['lm']
    full,info=refit_coefficients(pairs,y,langs)
    assert info['identifiable'] and info['rank']==21,info
    for j,language in enumerate(langs):
        full_rows.append({'mode':mode,'language':language,'alpha':float(full['alpha'][j]),
                          'beta':float(full['beta'][j]),'theta':float(full['theta'][j])})
    for omitted in langs:
        remaining=[l for l in langs if l!=omitted]
        train_names=[p for p in pairs if omitted not in p.split('__')]
        train_y=np.array([y[pairs.index(p)] for p in train_names],float)
        fit,rank_info=refit_coefficients(train_names,train_y,remaining)
        row={'mode':mode,'omitted_language':omitted,'n_languages_fit':len(remaining),
             'n_directed_edges_fit':len(train_names),**rank_info}
        if fit is None:
            deletion_rows.append(row)
            continue
        idx=[langs.index(l) for l in remaining]
        for name in ('alpha','beta','theta'):
            reference=full[name][idx]-full[name][idx].mean()
            refitted=fit[name]
            delta=refitted-reference
            row[f'{name}_spearman_remaining']=rank_corr(reference,refitted)
            row[f'{name}_rms_shift']=float(np.sqrt(np.mean(delta**2)))
            row[f'{name}_max_abs_shift']=float(np.max(np.abs(delta)))
            row[f'{name}_shift_over_full_sd']=float(np.sqrt(np.mean(delta**2))/np.std(reference))
            for j,language in enumerate(remaining):
                coefficient_rows.append({'mode':mode,'omitted_language':omitted,
                   'retained_language':language,'coefficient':name,
                   'full_recentered':float(reference[j]),'refit':float(refitted[j]),
                   'shift':float(delta[j])})
        deletion_rows.append(row)

full_coefficients=pd.DataFrame(full_rows)
deletion_summary=pd.DataFrame(deletion_rows)
deletion_coefficients=pd.DataFrame(coefficient_rows)
assert deletion_summary.groupby('mode').size().eq(11).all()
assert deletion_summary.identifiable.all(),'One deletion makes the reduced model unidentifiable; inspect table.'
full_coefficients.to_csv(OUTPUT/'full_language_coefficients.csv',index=False)
deletion_summary.to_csv(OUTPUT/'leave_one_language_out_summary.csv',index=False)
deletion_coefficients.to_csv(OUTPUT/'leave_one_language_out_coefficients.csv',index=False)
display(deletion_summary[['mode','omitted_language','n_directed_edges_fit','rank',
    'alpha_spearman_remaining','beta_spearman_remaining','theta_spearman_remaining',
    'alpha_shift_over_full_sd','beta_shift_over_full_sd']])


# %% original code cell 4; notebook cell 9

import matplotlib.pyplot as plt
MODEL_ORDER=['both','source','target','mean']
MODEL_LABEL={'both':'source + target','source':'source only',
             'target':'target only','mean':'mean only'}
def draw_panels(ax_beta,ax_r2,coefficients,r2_table,title):
    order=sorted(coefficients.language.unique())
    sub=coefficients[coefficients['mode']=='teacher_forced'].set_index('language').loc[order]
    x=np.arange(len(order))
    ax_beta.bar(x,sub.beta.to_numpy(float),color='#2457A6')
    ax_beta.axhline(0,color='0.5',linewidth=.8)
    ax_beta.set_xticks(x,order,rotation=55,ha='right',fontsize=8)
    ax_beta.set_ylabel('Target coefficient β (attention mass)')
    ax_beta.set_title(f'{title}: teacher-forced β')
    sub=r2_table[r2_table['mode']=='teacher_forced'].set_index('model').loc[MODEL_ORDER]
    x=np.arange(len(MODEL_ORDER))
    ax_r2.bar(x,sub.directed_LOEO_R2_centered.to_numpy(float),color='#D47B22')
    ax_r2.axhline(0,color='0.5',linewidth=.8)
    ax_r2.set_xticks(x,[MODEL_LABEL[m] for m in MODEL_ORDER],rotation=18,ha='right',fontsize=8)
    ax_r2.set_ylabel('Held-out-dyad directed R²')
    ax_r2.set_title(f'{title}: four additive model forms')
fig,axes=plt.subplots(1,2,figsize=(12.5,4.4),layout='constrained')
draw_panels(axes[0],axes[1],full_coefficients,loeo_summary,
            {'pilot':'IndicTrans2 pilot (fp16, n=250)',
             'holdout':'IndicTrans2 holdout (fp32, n=450)',
             'nllb':'NLLB confirmatory (fp32, n=450)'}[PRIMARY])
fig.savefig(OUTPUT/'beta_and_four_model_R2_TF.png',dpi=240)
fig.savefig(OUTPUT/'beta_and_four_model_R2_TF.pdf')
display(fig)
plt.close(fig)


# %% original code cell 5; notebook cell 11

origin=('original fp16 250-meaning collector' if PRIMARY=='pilot'
        else 'new fp32 450-meaning teacher-forced baseline')
statement=(
    f'This CPU-only teacher-forced analysis uses the {origin} for {PRIMARY}. '
    f'Input files, SHA-256 hashes, run settings, sample manifest and exclusions '
    f'are recorded in input_provenance.json. The old pilot has 250 meanings. '
    f'The fp32 holdout and NLLB baselines have 450 meanings across all 40 directions; '
    f'the remaining 300 intervention meanings are not a complete 40-direction baseline. '
    f'All 20 reciprocal dyads use complete paired meaning support. Source/target '
    f'coefficients are centered for identifiability. Each language deletion compares '
    f'only the ten retained-language coefficients. The figure reports teacher-forced '
    f'β and leave-one-dyad-out R² for four model forms. The original NLLB free-generation '
    f'capped/omitted rows do not enter this teacher-forced analysis. The notebooks, '
    f'builder and README are supplied in a local code archive, not published publicly. '
    f'Reproduction requires access to the saved collector backups; FLORES+ text is not redistributed.'
)
(OUTPUT/'reproducibility_statement.txt').write_text(statement+'\n',encoding='utf-8')
provenance={'study':PRIMARY,'cpu_only':True,'mode':'teacher_forced','precision':'fp16' if PRIMARY=='pilot' else 'fp32',
  'n_meanings':EXPECTED_N[PRIMARY],'n_directions':40,'n_reciprocal_dyads':20,
  'input_sha256':input_hashes,'run_settings':run_settings,
  'source_root':str(root),'figure_model_order':MODEL_ORDER,
  'result_files':['full_language_coefficients.csv','leave_one_language_out_summary.csv',
                  'leave_one_language_out_coefficients.csv','directed_additive_loeo_summary.csv',
                  'beta_and_four_model_R2_TF.png','beta_and_four_model_R2_TF.pdf']}
(OUTPUT/'input_provenance.json').write_text(json.dumps(provenance,indent=2,default=str),encoding='utf-8')
print(statement)
print('Saved CPU results and figures to',OUTPUT)


# %% original code cell 6; notebook cell 13

all_roots={name:Path('/content/drive/MyDrive/indic_40_lolo_mainfigure_cpu')/name
           for name in ('pilot','holdout','nllb')}
missing=[str(p) for p in all_roots.values() if not (p/'full_language_coefficients.csv').exists()
                                           or not (p/'directed_additive_loeo_summary.csv').exists()]
if missing:
    print('Combined main-text figure waits for these CPU notebooks:',missing)
else:
    fig,axes=plt.subplots(3,2,figsize=(13,12.5),layout='constrained')
    for j,(name,title) in enumerate([('pilot','IndicTrans2 pilot (fp16, n=250)'),
                                     ('holdout','IndicTrans2 holdout (fp32, n=450)'),
                                     ('nllb','NLLB confirmatory (fp32, n=450)')]):
        a=pd.read_csv(all_roots[name]/'full_language_coefficients.csv')
        b=pd.read_csv(all_roots[name]/'directed_additive_loeo_summary.csv')
        assert set(a.language)==set(langs) and set(b.model)==set(MODEL_ORDER)
        assert set(a['mode'])=={'teacher_forced'} and set(b['mode'])=={'teacher_forced'}
        draw_panels(axes[j,0],axes[j,1],a,b,title)
    fig.savefig(OUTPUT.parent/'main_text_beta_and_four_model_R2_TF_three_studies.png',dpi=240)
    fig.savefig(OUTPUT.parent/'main_text_beta_and_four_model_R2_TF_three_studies.pdf')
    display(fig)
    plt.close(fig)
    print('Combined teacher-forced main-text figure saved to',OUTPUT.parent)

