# Original executed notebook code, in original cell order.
# Requires Colab/IPython and original inputs; see docs/VALIDATION_EXPERIMENTS.md.

# %% original code cell 0; notebook cell 1
# 1. Mount Drive and configure paths. No original files are changed.
import os,sys,subprocess,queue,threading
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from google.colab import drive,userdata
from IPython.display import display
drive.mount('/content/drive')
FP32_ROOT=Path('/content/drive/MyDrive/fp32_tag_intervention')
OUTPUT_ROOT=Path('/content/drive/MyDrive/indic_40_enhancements/E3_madlad_fp32_v1')
STAGING=Path('/content/e3_madlad_staging');STAGING.mkdir(parents=True,exist_ok=True)
WORKER_ENV=Path('/content/e3_madlad_python312')
MODEL_CACHE=Path('/content/drive/MyDrive/indic_40_model_cache/madlad400_3b_fa184c675da0')
N_BOOT=10_000
print('Existing fp32 input:',FP32_ROOT)
print('New E3 output:',OUTPUT_ROOT)
print('New collection: 18,000 fp32 TF cells on MADLAD only. All checkpoints go to Drive.')


# %% original code cell 1; notebook cell 2
"""E1 generalization from saved, complete fp32 teacher-forced panels. CPU only."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_ORDER = ('mean', 'source', 'target', 'both', 'direction_mean')
PROTOCOLS = ('unseen_meanings', 'unseen_dyad_shared_meanings', 'unseen_meanings_and_dyad')
SCALES = ('individual_rows', 'direction_means')
METRICS = ('R2_test_centered', 'R2_training_mean_reference', 'MAE', 'RMSE')

def progress(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def sha256_file(path):
    with Path(path).open('rb') as stream:
        return sha256_bytes(stream.read())

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def atomic_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False); tmp.replace(path)

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays); tmp.replace(path)

def freeze_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != obj:
            raise ValueError(f'Existing frozen settings differ: {path}. Use a new output directory for a changed protocol.')
    else:
        atomic_json(path, obj)

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

def truth_values(series):
    return series.astype(str).str.strip().str.lower().isin(('true', '1', '1.0'))

# E1_SECTION: INPUTS
def load_fp32_panel(root, study, expected, original_split):
    """Hash-verify complete TF rows. No tensor, corpus download, or imputation."""
    run = 'holdout' if study == 'it2' else 'confirm'
    folder = Path(root) / run / 'baseline_tf'
    require(folder.is_dir(), f'Missing baseline folder: {folder}. Mount Drive or edit FP32_ROOT; do not substitute intervention/old fp16 rows.')
    files = sorted(folder.glob('*.csv'))
    require(len(files) == 40, f'{study}: expected exactly 40 baseline CSVs; found {len(files)} at {folder}')
    expected_files = expected['files'][study]
    require({p.name for p in files} == set(expected_files), f'{study}: baseline filenames differ from the audited completed run')
    lock_path = Path(root) / run / 'run_lock.json'
    require(lock_path.is_file(), f'Missing original run lock: {lock_path}')
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    require(lock == expected['run_locks'][study], f'{study}: original run lock differs from the audited fp32 run')
    allowed = set(map(int, original_split['baseline450']))
    rows = []; hashes = {}; started = time.perf_counter()
    required = {'sentence_id', 'corpus_sentence_id', 'source', 'tag', 'text', 'eligible', 'lm', 'lexical', 'closure_maxerr', 'source_len', 'target_len'}
    for i, path in enumerate(files, 1):
        raw = path.read_bytes(); digest = sha256_bytes(raw)
        require(digest == expected_files[path.name], f'{study}: input hash changed: {path.name}. Stop and inspect provenance.')
        hashes[path.name] = digest
        # Strings preserve the original corpus identity; numeric sentence_id is local to the 750-meaning sample.
        import io
        frame = pd.read_csv(io.BytesIO(raw), dtype={'corpus_sentence_id': str})
        require(required <= set(frame), f'{path.name}: missing columns {sorted(required-set(frame))}')
        require(len(frame) == 450, f'{path.name}: expected 450 rows, got {len(frame)}')
        require(frame[list(required)].notna().all().all(), f'{path.name}: missing identity/attention values')
        require(truth_values(frame.eligible).all(), f'{path.name}: ineligible TF row; no silent filtering is permitted')
        if 'reason' in frame:
            require(frame.reason.fillna('').astype(str).str.strip().eq('').all(), f'{path.name}: nonempty exclusion reason')
        frame['sentence_id'] = pd.to_numeric(frame.sentence_id, errors='raise').astype(int)
        require(set(frame.sentence_id) == allowed and not frame.sentence_id.duplicated().any(), f'{path.name}: meaning set differs from baseline450')
        require(frame.tag.eq(frame.text).all(), f'{path.name}: tag/text mismatch; intervention rows are forbidden')
        require(frame.source.nunique() == frame.tag.nunique() == 1, f'{path.name}: mixed directions')
        pair = str(frame.source.iloc[0]) + '__' + str(frame.tag.iloc[0])
        require(not frame.source.eq(frame.tag).any(), f'{path.name}: self direction')
        require(path.stem == pair + '__' + str(frame.tag.iloc[0]), f'{path.name}: filename/direction mismatch')
        components = ['source_tag', 'target_tag', 'eos', 'unknown', 'punctuation', 'lexical'] if study == 'it2' else ['source_code', 'eos', 'punctuation', 'lexical']
        require(set(components) <= set(frame), f'{path.name}: component columns differ from the original model protocol')
        mass = frame[components].apply(pd.to_numeric, errors='raise').to_numpy(float)
        require(np.isfinite(mass).all() and (mass >= -1e-7).all() and (mass <= 1+1e-5).all(), f'{path.name}: invalid component masses')
        require(np.allclose(mass.sum(1), 1, atol=1e-5, rtol=0), f'{path.name}: component closure failed')
        require(np.isfinite(frame.lm).all() and frame.lm.between(0, 1).all(), f'{path.name}: invalid lexical mass')
        require(np.allclose(frame.lm, frame.lexical, atol=1e-7, rtol=0), f'{path.name}: LM differs from lexical share')
        require((frame.closure_maxerr < 1e-5).all() and (frame.source_len > 0).all() and (frame.target_len > 0).all(), f'{path.name}: recorded closure/length gate failed')
        frame['pair'] = pair
        rows.append(frame[['sentence_id', 'corpus_sentence_id', 'pair', 'lm']])
        if i == 1 or i % 5 == 0 or i == len(files):
            progress(f'{study}: validated CSV {i}/40; {time.perf_counter()-started:.1f}s')
    frame = pd.concat(rows, ignore_index=True)
    require(len(frame) == 18000 and not frame.duplicated(['sentence_id', 'pair']).any(), f'{study}: incomplete/duplicate panel')
    require(frame.groupby('sentence_id').pair.nunique().eq(40).all(), f'{study}: an incomplete meaning block')
    require(frame.groupby('sentence_id').corpus_sentence_id.nunique().eq(1).all(), f'{study}: corpus identities disagree across directions')
    manifest = pd.read_csv(Path(root)/'sample_identity.csv', usecols=['pair', 'sentence_id', 'corpus_sentence_id'], dtype={'corpus_sentence_id': str})
    reference = manifest[manifest.sentence_id.isin(allowed)]
    joined = frame.merge(reference, on=['pair', 'sentence_id'], how='left', validate='one_to_one', suffixes=('', '_manifest'))
    require(joined.corpus_sentence_id.eq(joined.corpus_sentence_id_manifest).all(), f'{study}: corpus IDs disagree with the hash-verified original manifest')
    pairs = sorted(frame.pair.unique()); ids = sorted(allowed)
    languages = sorted({l for p in pairs for l in p.split('__')})
    require(len(languages) == 11, f'{study}: expected 11 languages')
    pivot = frame.pivot(index='sentence_id', columns='pair', values='lm').reindex(index=ids, columns=pairs)
    Y = pivot.to_numpy(float)
    require(Y.shape == (450, 40) and np.isfinite(Y).all(), f'{study}: invalid matrix')
    identity = frame[['sentence_id', 'corpus_sentence_id']].drop_duplicates().set_index('sentence_id').loc[ids].corpus_sentence_id.astype(str).tolist()
    return {'study': study, 'Y': Y, 'ids': ids, 'pairs': pairs, 'languages': languages,
            'corpus_ids': identity, 'csv_sha256': hashes, 'run_lock_sha256': sha256_file(lock_path)}

def validate_root(root, expected, original_split_sha256):
    root = Path(root)
    for name, digest in [('frozen_meaning_split.json', original_split_sha256), ('sample_identity.csv', expected['sample_sha256'])]:
        path = root / name
        require(path.is_file(), f'Missing original input manifest: {path}')
        require(sha256_file(path) == digest, f'Original manifest hash differs: {path}')
    split = json.loads((root/'frozen_meaning_split.json').read_text(encoding='utf-8'))
    require(len(split['baseline450']) == 450 and len(split['intervention300']) == 300, 'Original 450/300 split is malformed')
    require(not set(split['baseline450']) & set(split['intervention300']), 'Original baseline and intervention overlap')
    return split

# E1_SECTION: DESIGN
def reverse(pair):
    a, b = pair.split('__'); return b + '__' + a

def design(pairs, languages, method):
    n = len(languages)
    if method == 'direction_mean':
        raise ValueError('Direction mean is represented by identity; unseen-dyad prediction is not estimable')
    blocks = [np.ones((len(pairs), 1))]
    for side, include in [(0, method in ('source', 'both')), (1, method in ('target', 'both'))]:
        if include:
            block = np.full((len(pairs), n), -1/n)
            for i, p in enumerate(pairs): block[i, languages.index(p.split('__')[side])] += 1
            blocks.append(block)
    return np.column_stack(blocks)

def graph_connected(pairs, languages):
    adjacency = {l: set() for l in languages}
    for p in pairs:
        a, b = p.split('__'); adjacency[a].add(b); adjacency[b].add(a)
    seen = {languages[0]}; stack = [languages[0]]
    while stack:
        for neighbor in adjacency[stack.pop()] - seen:
            seen.add(neighbor); stack.append(neighbor)
    return len(seen) == len(languages)

def build_operators(pairs, languages):
    """Prediction matrices exactly refit OLS to each resampled training panel.

    Meaning resampling preserves the same count for every direction, so OLS on
    pooled rows equals OLS on resampled direction means. Design/pseudoinverses
    are fixed by the graph; coefficients/predictions change on every draw.
    """
    require(len(pairs) == 40 and len(set(pairs)) == 40, 'Expected 40 unique directions')
    require(all(reverse(p) in pairs for p in pairs), 'Missing reciprocal direction')
    dyads = [p for p in pairs if p < reverse(p)]
    require(len(dyads) == 20, 'Expected 20 dyads')
    full = {}; held = {}; audits = []
    expected_rank = {'mean': 1, 'source': len(languages), 'target': len(languages), 'both': 2*len(languages)-1}
    for method in MODEL_ORDER:
        if method == 'direction_mean':
            full[method] = np.eye(len(pairs)); continue
        X = design(pairs, languages, method)
        require(np.linalg.matrix_rank(X) == expected_rank[method], f'Full {method} design not identifiable')
        full[method] = X @ np.linalg.pinv(X)
        H = np.zeros((len(pairs), len(pairs)))
        for dyad in dyads:
            test = [pairs.index(dyad), pairs.index(reverse(dyad))]
            train = [i for i in range(len(pairs)) if i not in test]
            connected = graph_connected([pairs[i] for i in train], languages)
            rank = int(np.linalg.matrix_rank(X[train]))
            audits.append({'dyad': dyad, 'method': method, 'connected': connected,
                           'training_directions': len(train), 'rank': rank, 'expected_rank': expected_rank[method],
                           'identifiable': rank == expected_rank[method]})
            require(connected and rank == expected_rank[method], f'Dyad {dyad}: {method} fold invalid; rank={rank}, connected={connected}')
            H[np.ix_(test, train)] = X[test] @ np.linalg.pinv(X[train])
            require(np.all(H[np.ix_(test, test)] == 0), 'Held-dyad labels leaked into prediction operator')
        held[method] = H
    return {'full': full, 'held': held, 'dyads': dyads}, pd.DataFrame(audits)

def split_indices(panel, split):
    lookup = {v: i for i, v in enumerate(panel['ids'])}
    require(set(split['train']) | set(split['validation']) | set(split['test']) == set(panel['ids']), 'E1 split does not cover exactly baseline450')
    pieces = [set(split[k]) for k in ('train', 'validation', 'test')]
    require(not pieces[0]&pieces[1] and not pieces[0]&pieces[2] and not pieces[1]&pieces[2], 'E1 meaning leakage')
    require([len(x) for x in pieces] == [315, 67, 68], 'E1 split sizes changed')
    return {k: np.array([lookup[i] for i in split[k]], int) for k in ('train', 'validation', 'test')}

def fit_coefficients(panel, train_Y, operators):
    rows = []; mean = train_Y.mean(0); pairs = panel['pairs']; langs = panel['languages']; n = len(langs)
    specifications = [('all_directions', None, list(range(40)))]
    specifications += [('held_dyad', p, [i for i, q in enumerate(pairs) if q not in {p, reverse(p)}]) for p in operators['dyads']]
    for protocol, dyad, keep in specifications:
        for method in ('mean', 'source', 'target', 'both'):
            coef = np.linalg.pinv(design([pairs[i] for i in keep], langs, method)) @ mean[keep]
            rows.append({'study': panel['study'], 'fit': protocol, 'held_dyad': dyad, 'method': method,
                         'term': 'intercept', 'language': '', 'coefficient': float(coef[0])})
            at = 1
            for role, enabled in [('source', method in ('source','both')), ('target', method in ('target','both'))]:
                if enabled:
                    for language, value in zip(langs, coef[at:at+n]):
                        rows.append({'study': panel['study'], 'fit': protocol, 'held_dyad': dyad, 'method': method,
                                     'term': role, 'language': language, 'coefficient': float(value)})
                    at += n
    return pd.DataFrame(rows)

# E1_SECTION: SCORES
def safe_ratio_r2(sse, denominator):
    return np.divide(sse, denominator, out=np.full(np.shape(sse), np.nan, float), where=denominator > 1e-15) * -1 + 1

def batch_scores(prediction, Y_eval, weights, reference_prediction):
    """Metrics over complete meaning blocks; predictions are constant per direction.

    R2_test_centered uses the current evaluation grand mean. The training-mean
    reference uses the fitted mean-only predictor, excluding the held dyad in
    every LODO fold. Both reference definitions are reported explicitly.
    """
    P = np.asarray(prediction, float); W = np.asarray(weights, float)
    size = W.sum(1, keepdims=True); require((size > 0).all(), 'Empty bootstrap evaluation block')
    mu = W @ Y_eval / size; second = W @ (Y_eval**2) / size
    row_sse = np.maximum(np.mean(second - 2*P*mu + P**2, axis=1), 0)
    row_sst = np.maximum(second.mean(1) - mu.mean(1)**2, 0)
    row_reference_sse = np.maximum(np.mean(second - 2*reference_prediction*mu + reference_prediction**2, axis=1), 0)
    absolute = np.abs(Y_eval[None, :, :] - P[:, None, :])
    row_mae = np.einsum('btd,bt->b', absolute, W, optimize=True) / (size[:, 0]*Y_eval.shape[1])
    direction_sse = np.mean((mu-P)**2, axis=1)
    direction_sst = np.mean((mu-mu.mean(1, keepdims=True))**2, axis=1)
    direction_reference_sse = np.mean((mu-reference_prediction)**2, axis=1)
    return np.stack([
        np.column_stack([safe_ratio_r2(row_sse,row_sst), safe_ratio_r2(row_sse,row_reference_sse), row_mae, np.sqrt(row_sse)]),
        np.column_stack([safe_ratio_r2(direction_sse,direction_sst), safe_ratio_r2(direction_sse,direction_reference_sse), np.mean(np.abs(mu-P),axis=1), np.sqrt(direction_sse)])
    ], axis=1)

def protocol_data(train_Y, test_Y, operators):
    return {
        'unseen_meanings': (test_Y, operators['full']),
        'unseen_dyad_shared_meanings': (train_Y, operators['held']),
        'unseen_meanings_and_dyad': (test_Y, operators['held'])}

def point_evaluation(panel, split, operators, evaluation_partition='test', predictions=True):
    indices = split_indices(panel, split); train_Y = panel['Y'][indices['train']]
    eval_Y = panel['Y'][indices[evaluation_partition]]
    mean_train = train_Y.mean(0); summaries=[]; perdyad=[]; frames=[]
    for protocol, (Y, Hs) in protocol_data(train_Y, eval_Y, operators).items():
        if evaluation_partition == 'validation' and protocol == 'unseen_dyad_shared_meanings': continue
        identity_indices = indices['train'] if protocol == 'unseen_dyad_shared_meanings' else indices[evaluation_partition]
        reference = (Hs['mean'] @ mean_train)[None, :]
        prediction_table = {'study': panel['study'], 'protocol': protocol,
            'sentence_id': np.repeat(np.array(panel['ids'])[identity_indices], 40),
            'corpus_sentence_id': np.repeat(np.array(panel['corpus_ids'])[identity_indices], 40),
            'pair': np.tile(panel['pairs'], len(Y)), 'observed_lm': Y.reshape(-1)}
        for method in MODEL_ORDER:
            if method not in Hs:
                for scale in SCALES:
                    summaries.append({'study':panel['study'],'partition':evaluation_partition,'protocol':protocol,'method':method,'scale':scale,
                        'available':False,'reason':'Unrestricted direction mean cannot estimate an unseen dyad without an extra prediction rule.'})
                continue
            P = Hs[method] @ mean_train
            scores = batch_scores(P[None,:], Y, np.ones((1,len(Y))), reference)[0]
            for si,scale in enumerate(SCALES):
                summaries.append({'study':panel['study'],'partition':evaluation_partition,'protocol':protocol,'method':method,'scale':scale,
                    'available':True,'reason':'','evaluation_meanings':len(Y),'evaluation_rows':int(Y.size),
                    **{metric:float(scores[si,mi]) for mi,metric in enumerate(METRICS)}})
            prediction_table['pred_'+method] = np.tile(P, len(Y))
            for dyad in operators['dyads']:
                keep = [panel['pairs'].index(dyad), panel['pairs'].index(reverse(dyad))]
                error = Y[:,keep] - P[keep]; direction_error = Y[:,keep].mean(0)-P[keep]
                for scale,err in [('individual_rows',error),('direction_means',direction_error)]:
                    perdyad.append({'study':panel['study'],'partition':evaluation_partition,'protocol':protocol,'dyad':dyad,
                        'method':method,'scale':scale,'evaluation_meanings':len(Y),'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt(np.mean(err**2)))})
        if predictions:frames.append(pd.DataFrame(prediction_table))
    return pd.DataFrame(summaries),pd.DataFrame(perdyad),pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

# E1_SECTION: BOOTSTRAP
def bootstrap_evaluation(panel, split, operators, n_boot, seed, batch_size, checkpoint, fingerprint):
    idx = split_indices(panel,split); train_Y=panel['Y'][idx['train']]; test_Y=panel['Y'][idx['test']]
    specs=protocol_data(train_Y,test_Y,operators)
    keys=[(protocol,method) for protocol,(_,Hs) in specs.items() for method in MODEL_ORDER if method in Hs]
    samples=np.full((n_boot,len(keys),len(SCALES),len(METRICS)),np.nan)
    rng=np.random.default_rng(seed); completed=0; draw_chain=''
    if Path(checkpoint).is_file():
        with np.load(checkpoint,allow_pickle=False) as saved:
            require(str(saved['fingerprint'].item())==fingerprint,'Bootstrap checkpoint settings/input hash changed')
            require(saved['samples'].shape==samples.shape,'Bootstrap checkpoint shape changed')
            samples[:]=saved['samples'];completed=int(saved['completed'].item())
            rng.bit_generator.state=json.loads(str(saved['rng_state'].item()))
            draw_chain=str(saved['draw_chain'].item())
        progress(f'{panel["study"]}: resumed bootstrap {completed:,}/{n_boot:,}')
    started=time.perf_counter()
    for start in range(completed,n_boot,batch_size):
        stop=min(start+batch_size,n_boot);b=stop-start
        Wtrain=rng.multinomial(len(train_Y),np.full(len(train_Y),1/len(train_Y)),size=b)
        Wtest=rng.multinomial(len(test_Y),np.full(len(test_Y),1/len(test_Y)),size=b)
        draw_chain=sha256_bytes(draw_chain.encode()+Wtrain.tobytes()+Wtest.tobytes())
        train_mean=Wtrain@train_Y/len(train_Y)
        for ki,(protocol,method) in enumerate(keys):
            Y,Hs=specs[protocol];weights=Wtrain if protocol=='unseen_dyad_shared_meanings' else Wtest
            P=train_mean@Hs[method].T;reference=train_mean@Hs['mean'].T
            samples[start:stop,ki]=batch_scores(P,Y,weights,reference)
        if start==completed or stop==n_boot or stop//1000>start//1000:
            atomic_npz(checkpoint,samples=samples,completed=np.array(stop),fingerprint=np.array(fingerprint),
                rng_state=np.array(json.dumps(rng.bit_generator.state)),draw_chain=np.array(draw_chain),
                keys=np.array(['|'.join(k) for k in keys]),scales=np.array(SCALES),metrics=np.array(METRICS))
            progress(f'{panel["study"]}: bootstrap {stop:,}/{n_boot:,}; {time.perf_counter()-started:.1f}s this session')
    return samples,keys,draw_chain

def add_intervals(point,samples,keys):
    result=point.copy()
    for i,row in result.iterrows():
        if not row.available:continue
        ki=keys.index((row.protocol,row.method));si=SCALES.index(row.scale)
        for mi,metric in enumerate(METRICS):
            values=samples[:,ki,si,mi];finite=values[np.isfinite(values)]
            lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
            result.loc[i,metric+'_ci_low']=lo;result.loc[i,metric+'_ci_high']=hi
            result.loc[i,metric+'_bootstrap_valid']=len(finite)
    return result

def paired_contrasts(point,samples,keys):
    rows=[]
    for protocol in PROTOCOLS:
        both=keys.index((protocol,'both'));target=keys.index((protocol,'target'))
        for si,scale in enumerate(SCALES):
            b=point[(point.protocol==protocol)&(point.method=='both')&(point.scale==scale)].iloc[0]
            t=point[(point.protocol==protocol)&(point.method=='target')&(point.scale==scale)].iloc[0]
            for mi,metric in enumerate(METRICS):
                sign=1 if metric.startswith('R2') else -1
                values=sign*(samples[:,both,si,mi]-samples[:,target,si,mi]);finite=values[np.isfinite(values)]
                lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                rows.append({'study':b.study,'protocol':protocol,'scale':scale,'contrast':'both_vs_target',
                    'metric':metric,'positive_means':'higher R2' if sign==1 else 'lower prediction error',
                    'estimate':sign*(b[metric]-t[metric]),'ci_low':lo,'ci_high':hi,'bootstrap_valid':len(finite)})
    return pd.DataFrame(rows)

# E1_SECTION: RUN
def run_study(panel, split, operators, output, n_boot, seed, batch_size, code_hash):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature={'study':panel['study'],'input_csv_sha256':panel['csv_sha256'],
        'split':split,'n_boot':n_boot,'seed':seed,'batch_size':batch_size,'code_sha256':code_hash}
    fingerprint=sha256_bytes(json.dumps(signature,sort_keys=True).encode())
    freeze_json(output/'analysis_lock.json',signature)
    done=output/'completed.json'
    if done.exists():
        stamp=json.loads(done.read_text())
        require(stamp['fingerprint']==fingerprint,'Existing completed result has a different signature')
        require(all((output/name).is_file() and sha256_file(output/name)==digest for name,digest in stamp['result_sha256'].items()),'A completed output is missing or modified')
        progress(f'{panel["study"]}: completed results verified; reusing saved analysis')
        return pd.read_csv(output/'test_metrics_with_CI.csv'),pd.read_csv(output/'paired_model_contrasts.csv')
    idx=split_indices(panel,split)
    atomic_csv(output/'training_coefficients.csv',fit_coefficients(panel,panel['Y'][idx['train']],operators))
    validation,validation_dyads,_=point_evaluation(panel,split,operators,'validation',False)
    atomic_csv(output/'validation_point_metrics.csv',validation)
    atomic_csv(output/'validation_per_dyad.csv',validation_dyads)
    point,dyads,predictions=point_evaluation(panel,split,operators,'test',True)
    atomic_csv(output/'test_point_metrics.csv',point);atomic_csv(output/'test_per_dyad.csv',dyads)
    atomic_csv(output/'held_out_row_predictions.csv',predictions)
    samples,keys,draw_chain=bootstrap_evaluation(panel,split,operators,n_boot,seed,batch_size,output/'bootstrap_checkpoint.npz',fingerprint)
    summary=add_intervals(point,samples,keys);contrasts=paired_contrasts(point,samples,keys)
    atomic_csv(output/'test_metrics_with_CI.csv',summary);atomic_csv(output/'paired_model_contrasts.csv',contrasts)
    names=['training_coefficients.csv','validation_point_metrics.csv','validation_per_dyad.csv',
        'test_point_metrics.csv','test_per_dyad.csv','held_out_row_predictions.csv',
        'test_metrics_with_CI.csv','paired_model_contrasts.csv','bootstrap_checkpoint.npz']
    atomic_json(done,{'fingerprint':fingerprint,'bootstrap_draw_chain':draw_chain,'bootstrap_completed':n_boot,
        'result_sha256':{name:sha256_file(output/name) for name in names}})
    progress(f'{panel["study"]}: E1 finished; saved to {output}')
    return summary,contrasts


# %% original code cell 2; notebook cell 4
# 2. Embedded immutable sources, model assets and design.
EXPECTED=json.loads('{"files": {"it2": {"asm_Beng__ben_Beng__ben_Beng.csv": "4634845e28c39d95e7c8130971d5cfd11747e3e158e2da156ab3571e94335e67", "asm_Beng__kan_Knda__kan_Knda.csv": "81c88e00168ca2259e6787e77135ebd436072a3645d06d16af4dc031965c6b46", "ben_Beng__asm_Beng__asm_Beng.csv": "6ebb30daf35a4991f9b8b2bede5290ec1dcd33c3b2abe319b7847e0deb7576de", "ben_Beng__kan_Knda__kan_Knda.csv": "30b6b14f9dc8d4bdd7d6c54585e339428621be1f4a2fe7ab165252c7bbe74ea7", "ben_Beng__mar_Deva__mar_Deva.csv": "3bbbcdf86d1226466072057efdee555b4b0b806f902cb613cd4363622a23b855", "ben_Beng__tam_Taml__tam_Taml.csv": "c40518929585fde76be7ba987261ca167f77ea88271117dcd5c2f40605506566", "guj_Gujr__mal_Mlym__mal_Mlym.csv": "ce14cd4c07ec11e8137aabfff61ea2738bbbee03039fe9273ac56ce409383c78", "guj_Gujr__mar_Deva__mar_Deva.csv": "f853aa655b03c6cf0f2446c9c0d3980e73d2909a87f36a881bde5d7efdb86756", "guj_Gujr__pan_Guru__pan_Guru.csv": "72a2cb9e92658689eafc36ce542117b3ff4fe5a82f68b3e369b480cfab0411c1", "hin_Deva__kan_Knda__kan_Knda.csv": "0563a0404498980d2c98c9a176195989bfbbf271631ca942be1456fbf3940500", "hin_Deva__mal_Mlym__mal_Mlym.csv": "35e5193ac09a23882035688bd2fce64dc546470d128e3a201f4e754df0fd0bda", "hin_Deva__ory_Orya__ory_Orya.csv": "1a7eaaefeb09bfbcc753d96f664b38d3ac7e1dcc5aa29f91db01880345ecdb93", "hin_Deva__tam_Taml__tam_Taml.csv": "bc1a5e084215749a262e753743f9a740f1d334ddd6c7ff7410c77b2628235301", "kan_Knda__asm_Beng__asm_Beng.csv": "a9ca741ff9a9d3de7b74607f8ee4f87b6e8086d0b245ab49565f3c6489191212", "kan_Knda__ben_Beng__ben_Beng.csv": "3621e04c235045fb9c910afe57a20676593726e0b1480aaad56d5d389a771297", "kan_Knda__hin_Deva__hin_Deva.csv": "6ca052a4d5837fba7ec001e2ecd32317257fb0913610a7d1bb11a1a3ff4c96f2", "kan_Knda__mal_Mlym__mal_Mlym.csv": "b02df5f356eb72d2c8a38d551c0eec3af90d1035ad7db83f981544c560e76650", "kan_Knda__mar_Deva__mar_Deva.csv": "84776eed18969cfee5a7b1823bf043999e0983b90d0a046f6ae9bcf018d9e06a", "kan_Knda__tam_Taml__tam_Taml.csv": "bfc457c1dc99189a070cae1bf517f2e2bbc7dc9b9517bf4333cdc7bb612e5b6c", "mal_Mlym__guj_Gujr__guj_Gujr.csv": "bb9e390740e851548d35f92b580c580904b3bce71a042a3d9ee65a8a2d4f8fcb", "mal_Mlym__hin_Deva__hin_Deva.csv": "7fa15dc5e9f999d6c6a7cd936f1e0c3eea3e890310b06424f69c21249b255645", "mal_Mlym__kan_Knda__kan_Knda.csv": "c2a67cdb13447efea979449c2b232b2fd3c1c87e3d68c0995e36d81034eb6569", "mal_Mlym__tam_Taml__tam_Taml.csv": "7a6de872b1df6d2145bc0c8ac15e16d9380e9e4c51fb4c7e49ff43891210669d", "mal_Mlym__tel_Telu__tel_Telu.csv": "ec7db98f64b79552351d435c8364bc1e0b4974f0a5d772532c98e1a5634c20e7", "mar_Deva__ben_Beng__ben_Beng.csv": "1342491709a2464c2a4d50757da6ca52b6383b95b95989456870f31e881945d8", "mar_Deva__guj_Gujr__guj_Gujr.csv": "64560ef6135751c1a5b30f98dc16b1b60f5414ff87bc248516fea5a3100c119a", "mar_Deva__kan_Knda__kan_Knda.csv": "27752e44123549dd39b4b3d6b39e8acb37d1d6f733bbbb54cf42ecc3d08387aa", "ory_Orya__hin_Deva__hin_Deva.csv": "2bcdf4c178b1fdf44bcdd5a3f1800e2355722dd21cb5c4621afe34a0e3cec981", "ory_Orya__tel_Telu__tel_Telu.csv": "668ba745204726558eac09e21b8bdc8a7ee14cbefa34c7a7ab3547f26727f651", "pan_Guru__guj_Gujr__guj_Gujr.csv": "f75cf9bcfd8954ebc0b5049577a15bc233f45a0abcb49cbaaaedfb02b517ed4a", "pan_Guru__tam_Taml__tam_Taml.csv": "84a06227916ef7155c3e8da7f832cd7a95961cf81524c143380b09c1063fd4d2", "tam_Taml__ben_Beng__ben_Beng.csv": "acd49c2d8d155f060391808ee76a17d2a4d6ae8c64ec2329d6361f763d0207e6", "tam_Taml__hin_Deva__hin_Deva.csv": "8db4da21c070a11e0e69441e3e070400101f72b102ed27e55fc317818e6cb18e", "tam_Taml__kan_Knda__kan_Knda.csv": "c2d1c55f4fc586deecf37da432c1433f92fd128cb275da18ebb84637961753b2", "tam_Taml__mal_Mlym__mal_Mlym.csv": "4cac02d01a1292ad5a9f52dc003632fc59ecde6cfd68df3ce1210dfacdc10400", "tam_Taml__pan_Guru__pan_Guru.csv": "50b8d0cafdd44ce91d21e0f17ecddd1269ccfda80c833a3aae6387fb2cfe45da", "tam_Taml__tel_Telu__tel_Telu.csv": "b76d1fade10dc79e5b0bcfe44618780a9ad4e72850f5582bf0d73353c2ce75bc", "tel_Telu__mal_Mlym__mal_Mlym.csv": "df3b7400b553b79c2a55d71b4dec630eb58e77df66b16bf1467393e62ccca932", "tel_Telu__ory_Orya__ory_Orya.csv": "b658578c14b9382c979b903143f373932ca1efc40599350910bc9cd5588ca73f", "tel_Telu__tam_Taml__tam_Taml.csv": "e53e5ccbba0ec047d6bb717a2a65982820ae8e8e9c3172270b27da2816687188"}, "nllb": {"asm_Beng__ben_Beng__ben_Beng.csv": "2064bc6293a2865309e16516071df5c59022b912e16c96fdab5fd4b283cff36e", "asm_Beng__kan_Knda__kan_Knda.csv": "b577608427097fb2a9a00ae64ecddbdd38266e690dc0813509500fa28f7a92a6", "ben_Beng__asm_Beng__asm_Beng.csv": "c151e1db1ddfd6fcbab4c87481f416fcb5cea4b18989d8717e1c35fa2bd453cd", "ben_Beng__kan_Knda__kan_Knda.csv": "b8cd209b383a759a6fe1fd8cc91ad5083dc0bbdb37e5c20d5997bfecedf9d904", "ben_Beng__mar_Deva__mar_Deva.csv": "bea2e02895ef5f8f10531916970e6121755ea6504c9234f3b9989163ba7fc6e0", "ben_Beng__tam_Taml__tam_Taml.csv": "34dd6305e12e258a82fbc0c2d6f2e623be73ac0df0736be104219c2d06d950f8", "guj_Gujr__mal_Mlym__mal_Mlym.csv": "0acbda7063180c28f70c5be357c1acbc0456771842c1206e15c802d55046610b", "guj_Gujr__mar_Deva__mar_Deva.csv": "743b82bbe6ae411cb78b7c4dcef64800c2373c73acaf58c5a390a7e02e076ff4", "guj_Gujr__pan_Guru__pan_Guru.csv": "f0e7f433bb0ee10ac03a95c6f4cf2e4b08b1b0e0ef1c1ba558a6499cf2d1de8d", "hin_Deva__kan_Knda__kan_Knda.csv": "1097f756f9fe0630594cec12c4ab5d501f25c53c9034fb0f743228f1149af268", "hin_Deva__mal_Mlym__mal_Mlym.csv": "18e4d46bbf17f256f7eda806ca18433c722084cebc72cb7dbeb5466c67f49050", "hin_Deva__ory_Orya__ory_Orya.csv": "f21b16f1e446b7536517a02b212e72796f0dcea09e5d339e521b50bb6b3e5767", "hin_Deva__tam_Taml__tam_Taml.csv": "5da204353dad25cac0925c7737bf4468dea22f09eafae485ac8c090272c48310", "kan_Knda__asm_Beng__asm_Beng.csv": "f9a543bc88d15281b36ea46195928dfa4303f07dd5c2b9b75cc2910596dcd18b", "kan_Knda__ben_Beng__ben_Beng.csv": "47702404293386f00e3bc6fb43df22bd64716e6cf250cc9ff2c44db0b98bfd28", "kan_Knda__hin_Deva__hin_Deva.csv": "6aabeed62e2aed01e293118e63834b53a2ad8c7432c8002ed2ae6c0f97703b3b", "kan_Knda__mal_Mlym__mal_Mlym.csv": "886c84e25998f1994666c2da333635dd767bc89e7c9ea4b336971fff951c4d17", "kan_Knda__mar_Deva__mar_Deva.csv": "813f21e381342c9a1e5a9bcc91e548e3760cb29708c2d263170f9c588215a575", "kan_Knda__tam_Taml__tam_Taml.csv": "46a9d60b3ed4ac6c3574e7572b5c6fe367edb8bbd9e0bfeeaf530f576ebfedbd", "mal_Mlym__guj_Gujr__guj_Gujr.csv": "609980fbfe47a4b3ba7426defe6a9544867d17a809acde707b2364a0bfbecf5f", "mal_Mlym__hin_Deva__hin_Deva.csv": "afa69e281e7b50335f4e4f216f76f57cf9db42791f8d1382e308d45aeff39570", "mal_Mlym__kan_Knda__kan_Knda.csv": "d7e378175aced0fae1d577750f52b13176bc2568a2892982b30972894e7e9f7b", "mal_Mlym__tam_Taml__tam_Taml.csv": "6c62596177c3a4bcdacd8c9f6bad05ebafd108b9843168c326528e0510a19b3a", "mal_Mlym__tel_Telu__tel_Telu.csv": "bfff8f20f5119253d905c1f9fc02c157d2f75b48584ded9d2bbf209191e163c7", "mar_Deva__ben_Beng__ben_Beng.csv": "ca28071d7a88b46fc228a0d6a69d1645f2e9ca04388685253922381cb0865dee", "mar_Deva__guj_Gujr__guj_Gujr.csv": "727d3e552df5a250577f13aeae48e696b7809f4b0de422f28557a8803f73c56b", "mar_Deva__kan_Knda__kan_Knda.csv": "3aa565d5c06332a91f3dab40ca1a36335c13ed812619a901cacf70f25e95f0f4", "ory_Orya__hin_Deva__hin_Deva.csv": "dcafc3a172acd79191107371f21019cbffed71b03e05960788ec7532697a8df6", "ory_Orya__tel_Telu__tel_Telu.csv": "3ccd6b7146e9dcf7dac782a726c5b082e56cc9c66cec59842081e202b81ef965", "pan_Guru__guj_Gujr__guj_Gujr.csv": "90236621d8121c30c315d2e621eb1e323c9d2242bf88449b147a61700dc16702", "pan_Guru__tam_Taml__tam_Taml.csv": "89e84bf644951fc9d2b9cd7989db5522cb17559a49a113e9fc8dc993e77546d9", "tam_Taml__ben_Beng__ben_Beng.csv": "ecfa082cbadaaf05fef587a246f213461e17c641070da62b6ecec43340f51548", "tam_Taml__hin_Deva__hin_Deva.csv": "37458e8f157e3b47f2cd71f7e87b100e18d12069489834e48ed2237e7ac8091c", "tam_Taml__kan_Knda__kan_Knda.csv": "292ba6bfdd962065032c97497861e5cc02318a5027cab7162a9cc91e879b7659", "tam_Taml__mal_Mlym__mal_Mlym.csv": "2bc67edf61d73b41edeb4d07adf90fad48b5bd2ffa335751cf8ca2663f3666d0", "tam_Taml__pan_Guru__pan_Guru.csv": "f6ca99718db5ccdad447e9d68dc65869012dbb7693026e57b50b14d8f373ca6e", "tam_Taml__tel_Telu__tel_Telu.csv": "3f3922a2af7a1b079f78588436b991aacc8da08cb814c3bd92afef386abfc5bd", "tel_Telu__mal_Mlym__mal_Mlym.csv": "a8bda76d64a618a89ebfb3dda4c60857fbb6b8771e023e4a5a2d0b03ca0c4e2f", "tel_Telu__ory_Orya__ory_Orya.csv": "30ae2f50cb771d050cf75524fa00dbaca65f8c968985e15b21caf2ba925b67f9", "tel_Telu__tam_Taml__tam_Taml.csv": "6c0f1cd62eff90c51d1c8ef632f3f86536fa50e2c27b1e88c1bedb40f51e79a5"}}, "run_locks": {"it2": {"model": "ai4bharat/indictrans2-indic-indic-dist-320M", "model_revision": "ffb7582b6d43791f1fb26b2153fc065f2e9ea575", "dataset": "openlanguagedata/flores_plus", "dataset_revision": "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06", "split_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "split_seed": 240902, "bootstrap_seed": 240903, "n_boot": 10000, "targets": ["pan_Guru", "mal_Mlym", "tam_Taml", "guj_Gujr", "hin_Deva", "ben_Beng"], "sources": ["kan_Knda", "tel_Telu", "ory_Orya", "mar_Deva"], "control": "urd_Arab", "generation_meanings": 150, "generation_ids_sha256": "a54f1be4a1069f3e0d7e08cbf7151bc05712a84306f63e3f1bf874776b084c32", "torch_version": "2.11.0+cu128", "transformers_version": "4.51.3"}, "nllb": {"model": "facebook/nllb-200-distilled-600M", "model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d", "dataset": "openlanguagedata/flores_plus", "dataset_revision": "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06", "split_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "split_seed": 240902, "bootstrap_seed": 240903, "n_boot": 10000, "targets": ["pan_Guru", "mal_Mlym", "tam_Taml", "guj_Gujr", "hin_Deva", "ben_Beng"], "sources": ["kan_Knda", "tel_Telu", "ory_Orya", "mar_Deva"], "control": "urd_Arab", "generation_meanings": 150, "generation_ids_sha256": "a54f1be4a1069f3e0d7e08cbf7151bc05712a84306f63e3f1bf874776b084c32", "torch_version": "2.11.0+cu128", "transformers_version": "4.51.3"}}, "sample_sha256": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c"}')
MEANING_SPLIT=json.loads('{"seed": 20261101, "train": [0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 26, 27, 30, 33, 42, 45, 47, 51, 54, 56, 57, 58, 63, 64, 65, 68, 70, 72, 73, 75, 78, 79, 80, 82, 83, 85, 89, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 104, 108, 111, 112, 113, 121, 122, 124, 125, 126, 127, 128, 132, 133, 135, 136, 138, 142, 147, 150, 153, 154, 157, 158, 162, 163, 164, 166, 167, 168, 170, 172, 174, 175, 177, 178, 181, 186, 188, 195, 196, 197, 198, 199, 201, 203, 205, 206, 208, 211, 214, 218, 222, 225, 226, 228, 231, 235, 236, 237, 241, 243, 246, 248, 249, 251, 254, 255, 257, 259, 261, 263, 264, 266, 268, 271, 276, 278, 279, 283, 284, 285, 288, 290, 295, 299, 302, 303, 307, 309, 313, 314, 316, 318, 319, 321, 325, 327, 328, 330, 336, 337, 341, 342, 344, 346, 348, 352, 355, 357, 358, 359, 365, 366, 369, 370, 372, 376, 377, 386, 388, 389, 395, 398, 399, 400, 402, 403, 404, 409, 411, 412, 413, 415, 422, 427, 432, 433, 434, 437, 440, 441, 444, 448, 452, 456, 457, 459, 463, 467, 468, 470, 471, 472, 479, 484, 486, 490, 492, 495, 498, 502, 503, 508, 509, 510, 517, 520, 523, 525, 527, 528, 534, 535, 537, 542, 544, 546, 548, 551, 552, 555, 559, 561, 563, 564, 565, 568, 571, 575, 577, 579, 580, 582, 583, 584, 586, 588, 591, 592, 596, 605, 606, 608, 610, 612, 613, 615, 616, 618, 627, 628, 630, 632, 633, 634, 636, 637, 638, 639, 643, 645, 648, 651, 654, 656, 657, 658, 660, 661, 663, 667, 668, 669, 673, 677, 682, 683, 684, 685, 687, 689, 690, 692, 693, 696, 701, 702, 706, 707, 708, 710, 711, 712, 717, 718, 719, 722, 724, 728, 735, 739, 740, 743, 745], "validation": [5, 19, 28, 32, 52, 61, 69, 86, 120, 140, 151, 185, 219, 227, 252, 256, 269, 273, 298, 300, 323, 329, 331, 332, 345, 367, 374, 378, 380, 383, 393, 414, 428, 439, 442, 447, 475, 494, 512, 515, 530, 541, 549, 585, 603, 604, 622, 623, 631, 647, 650, 652, 655, 670, 674, 675, 676, 678, 697, 698, 705, 731, 733, 734, 737, 741, 748], "test": [8, 17, 24, 37, 40, 41, 49, 71, 77, 81, 87, 88, 93, 99, 106, 117, 130, 156, 165, 176, 190, 194, 200, 204, 207, 229, 232, 234, 245, 247, 253, 267, 274, 275, 281, 291, 305, 306, 315, 347, 351, 368, 375, 387, 391, 419, 426, 464, 474, 501, 511, 538, 595, 597, 621, 641, 649, 672, 681, 699, 703, 713, 723, 725, 726, 742, 746, 747], "original_split_sha256": "9ed17cbd0b4b2edd97e79bf3cd074af5929ee6ea98552ad0965f58afdaef43ef", "original_sample_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "scope": "450 original fp32 baseline meanings only"}')
PROTOCOL=json.loads('{"experiment": "E3", "version": "1.0", "models": ["it2", "nllb", "madlad"], "new_model": {"repo": "google/madlad400-3b-mt", "revision": "fa184c675da0b5c9e1c8694fccd4e12e2d422094", "files": {"config.json": "cad399cab799b99409a6ec2d90d72552257c2bb752861261d2016691e0643e7c", "tokenizer_config.json": "641fc660745306dfb935f666a68f8bc10a44c39241cfb357be518fda8c09662d", "special_tokens_map.json": "7f79f1d5063d56c4b980eec0692f3c7429bdef335071d34e566bd00fd4b5e3e0", "spiece.model": "ef11ac9a22c7503492f56d48dce53be20e339b63605983e9f27d2cd0e0f3922c", "added_tokens.json": "d914176fd50bd7f565700006a31aa97b79d3ad17cee20c8e5ff2061d5cb74817", "model.safetensors": "66ff5f8fcaf92291da486fdfbd4d5233cec90e1359348a56e3172c978b3a76d4"}}, "meanings": "same 450 original baseline meanings; intervention300 unused", "directions": "same 40 directed values / 20 reciprocal dyads / 11 languages", "new_collection": "fp32 TF only; batch1; no quantization, autocast, TF32, offloading or translation generation", "inference": "reference labels shifted by T5; decoder start0; encoder prefix <2target-code>; no source language code", "target_rows": "reference lexical labels; exclude EOS/punctuation/unknown for MADLAD; original IT2/NLLB masks preserved", "source_partition": "MADLAD target tag / EOS / unknown / punctuation / lexical, mutually exclusive and exhaustive; blank SP pieces count as punctuation", "cross_model_unknown_caveat": "IT2 and MADLAD exclude unknown from lexical; original NLLB includes unknown. No unsupported correction to NLLB.", "exclusions": "no truncation or silent dropping; fail with saved context on empty support, excessive length or OOM", "length_safety_ceiling": 2048, "checkpoint": "25 rows; per-row CSV and per-head 32x16x5 component shares", "split": {"seed": 20261101, "train": [0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 26, 27, 30, 33, 42, 45, 47, 51, 54, 56, 57, 58, 63, 64, 65, 68, 70, 72, 73, 75, 78, 79, 80, 82, 83, 85, 89, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 104, 108, 111, 112, 113, 121, 122, 124, 125, 126, 127, 128, 132, 133, 135, 136, 138, 142, 147, 150, 153, 154, 157, 158, 162, 163, 164, 166, 167, 168, 170, 172, 174, 175, 177, 178, 181, 186, 188, 195, 196, 197, 198, 199, 201, 203, 205, 206, 208, 211, 214, 218, 222, 225, 226, 228, 231, 235, 236, 237, 241, 243, 246, 248, 249, 251, 254, 255, 257, 259, 261, 263, 264, 266, 268, 271, 276, 278, 279, 283, 284, 285, 288, 290, 295, 299, 302, 303, 307, 309, 313, 314, 316, 318, 319, 321, 325, 327, 328, 330, 336, 337, 341, 342, 344, 346, 348, 352, 355, 357, 358, 359, 365, 366, 369, 370, 372, 376, 377, 386, 388, 389, 395, 398, 399, 400, 402, 403, 404, 409, 411, 412, 413, 415, 422, 427, 432, 433, 434, 437, 440, 441, 444, 448, 452, 456, 457, 459, 463, 467, 468, 470, 471, 472, 479, 484, 486, 490, 492, 495, 498, 502, 503, 508, 509, 510, 517, 520, 523, 525, 527, 528, 534, 535, 537, 542, 544, 546, 548, 551, 552, 555, 559, 561, 563, 564, 565, 568, 571, 575, 577, 579, 580, 582, 583, 584, 586, 588, 591, 592, 596, 605, 606, 608, 610, 612, 613, 615, 616, 618, 627, 628, 630, 632, 633, 634, 636, 637, 638, 639, 643, 645, 648, 651, 654, 656, 657, 658, 660, 661, 663, 667, 668, 669, 673, 677, 682, 683, 684, 685, 687, 689, 690, 692, 693, 696, 701, 702, 706, 707, 708, 710, 711, 712, 717, 718, 719, 722, 724, 728, 735, 739, 740, 743, 745], "validation": [5, 19, 28, 32, 52, 61, 69, 86, 120, 140, 151, 185, 219, 227, 252, 256, 269, 273, 298, 300, 323, 329, 331, 332, 345, 367, 374, 378, 380, 383, 393, 414, 428, 439, 442, 447, 475, 494, 512, 515, 530, 541, 549, 585, 603, 604, 622, 623, 631, 647, 650, 652, 655, 670, 674, 675, 676, 678, 697, 698, 705, 731, 733, 734, 737, 741, 748], "test": [8, 17, 24, 37, 40, 41, 49, 71, 77, 81, 87, 88, 93, 99, 106, 117, 130, 156, 165, 176, 190, 194, 200, 204, 207, 229, 232, 234, 245, 247, 253, 267, 274, 275, 281, 291, 305, 306, 315, 347, 351, 368, 375, 387, 391, 419, 426, 464, 474, 501, 511, 538, 595, 597, 621, 641, 649, 672, 681, 699, 703, 713, 723, 725, 726, 742, 746, 747], "original_split_sha256": "9ed17cbd0b4b2edd97e79bf3cd074af5929ee6ea98552ad0965f58afdaef43ef", "original_sample_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "scope": "450 original fp32 baseline meanings only"}, "bootstrap": {"repetitions": 10000, "generalization_seed": 20261102, "allocation_seed": 20261106, "unit": "whole meanings; graph/languages fixed", "generalization": "E1 exact training-refit operators; independent train/test resampling", "allocation": "joint resampling on all450; percentile intervals; descriptive in-sample fits"}, "edge_null": "10000 independent permutations of signed dyad means; fixed graph; descriptive, not a sentence-noise null", "primary": "new-model unseen-meaning + unseen-dyad prediction and within-model allocation structure; report all results regardless of outcome", "limitations": "existing corpus and selected language graph; model language support does not guarantee accurate translation on every direction; attention is descriptive", "publication": "private local protocol freeze and release-ready code; no public registration or GitHub mutation required"}')
MODEL_SPEC=json.loads('{"repo": "google/madlad400-3b-mt", "revision": "fa184c675da0b5c9e1c8694fccd4e12e2d422094", "files": {"config.json": "cad399cab799b99409a6ec2d90d72552257c2bb752861261d2016691e0643e7c", "tokenizer_config.json": "641fc660745306dfb935f666a68f8bc10a44c39241cfb357be518fda8c09662d", "special_tokens_map.json": "7f79f1d5063d56c4b980eec0692f3c7429bdef335071d34e566bd00fd4b5e3e0", "spiece.model": "ef11ac9a22c7503492f56d48dce53be20e339b63605983e9f27d2cd0e0f3922c", "added_tokens.json": "d914176fd50bd7f565700006a31aa97b79d3ad17cee20c8e5ff2061d5cb74817", "model.safetensors": "66ff5f8fcaf92291da486fdfbd4d5233cec90e1359348a56e3172c978b3a76d4"}}')
LANG_CODES=json.loads('{"asm_Beng": "as", "ben_Beng": "bn", "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn", "mal_Mlym": "ml", "mar_Deva": "mr", "ory_Orya": "or", "pan_Guru": "pa", "tam_Taml": "ta", "tel_Telu": "te"}')
TAG_IDS=json.loads('{"asm_Beng": 12, "ben_Beng": 23, "guj_Gujr": 55, "hin_Deva": 58, "kan_Knda": 74, "mal_Mlym": 94, "mar_Deva": 97, "ory_Orya": 111, "pan_Guru": 112, "tam_Taml": 136, "tel_Telu": 139}')
E1_SOURCE='"""E1 generalization from saved, complete fp32 teacher-forced panels. CPU only."""\nimport hashlib\nimport json\nimport os\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n\nMODEL_ORDER = (\'mean\', \'source\', \'target\', \'both\', \'direction_mean\')\nPROTOCOLS = (\'unseen_meanings\', \'unseen_dyad_shared_meanings\', \'unseen_meanings_and_dyad\')\nSCALES = (\'individual_rows\', \'direction_means\')\nMETRICS = (\'R2_test_centered\', \'R2_training_mean_reference\', \'MAE\', \'RMSE\')\n\ndef progress(message):\n    print(time.strftime(\'%H:%M:%S\'), message, flush=True)\n\ndef sha256_bytes(raw):\n    return hashlib.sha256(raw).hexdigest()\n\ndef sha256_file(path):\n    with Path(path).open(\'rb\') as stream:\n        return sha256_bytes(stream.read())\n\ndef atomic_json(path, obj):\n    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + \'.tmp\')\n    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + \'\\n\', encoding=\'utf-8\')\n    tmp.replace(path)\n\ndef atomic_csv(path, frame):\n    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + \'.tmp\')\n    frame.to_csv(tmp, index=False); tmp.replace(path)\n\ndef atomic_npz(path, **arrays):\n    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(\'.tmp.npz\')\n    np.savez_compressed(tmp, **arrays); tmp.replace(path)\n\ndef freeze_json(path, obj):\n    path = Path(path)\n    if path.exists():\n        if json.loads(path.read_text(encoding=\'utf-8\')) != obj:\n            raise ValueError(f\'Existing frozen settings differ: {path}. Use a new output directory for a changed protocol.\')\n    else:\n        atomic_json(path, obj)\n\ndef require(condition, message):\n    if not bool(condition):\n        raise ValueError(message)\n\ndef truth_values(series):\n    return series.astype(str).str.strip().str.lower().isin((\'true\', \'1\', \'1.0\'))\n\n# E1_SECTION: INPUTS\ndef load_fp32_panel(root, study, expected, original_split):\n    """Hash-verify complete TF rows. No tensor, corpus download, or imputation."""\n    run = \'holdout\' if study == \'it2\' else \'confirm\'\n    folder = Path(root) / run / \'baseline_tf\'\n    require(folder.is_dir(), f\'Missing baseline folder: {folder}. Mount Drive or edit FP32_ROOT; do not substitute intervention/old fp16 rows.\')\n    files = sorted(folder.glob(\'*.csv\'))\n    require(len(files) == 40, f\'{study}: expected exactly 40 baseline CSVs; found {len(files)} at {folder}\')\n    expected_files = expected[\'files\'][study]\n    require({p.name for p in files} == set(expected_files), f\'{study}: baseline filenames differ from the audited completed run\')\n    lock_path = Path(root) / run / \'run_lock.json\'\n    require(lock_path.is_file(), f\'Missing original run lock: {lock_path}\')\n    lock = json.loads(lock_path.read_text(encoding=\'utf-8\'))\n    require(lock == expected[\'run_locks\'][study], f\'{study}: original run lock differs from the audited fp32 run\')\n    allowed = set(map(int, original_split[\'baseline450\']))\n    rows = []; hashes = {}; started = time.perf_counter()\n    required = {\'sentence_id\', \'corpus_sentence_id\', \'source\', \'tag\', \'text\', \'eligible\', \'lm\', \'lexical\', \'closure_maxerr\', \'source_len\', \'target_len\'}\n    for i, path in enumerate(files, 1):\n        raw = path.read_bytes(); digest = sha256_bytes(raw)\n        require(digest == expected_files[path.name], f\'{study}: input hash changed: {path.name}. Stop and inspect provenance.\')\n        hashes[path.name] = digest\n        # Strings preserve the original corpus identity; numeric sentence_id is local to the 750-meaning sample.\n        import io\n        frame = pd.read_csv(io.BytesIO(raw), dtype={\'corpus_sentence_id\': str})\n        require(required <= set(frame), f\'{path.name}: missing columns {sorted(required-set(frame))}\')\n        require(len(frame) == 450, f\'{path.name}: expected 450 rows, got {len(frame)}\')\n        require(frame[list(required)].notna().all().all(), f\'{path.name}: missing identity/attention values\')\n        require(truth_values(frame.eligible).all(), f\'{path.name}: ineligible TF row; no silent filtering is permitted\')\n        if \'reason\' in frame:\n            require(frame.reason.fillna(\'\').astype(str).str.strip().eq(\'\').all(), f\'{path.name}: nonempty exclusion reason\')\n        frame[\'sentence_id\'] = pd.to_numeric(frame.sentence_id, errors=\'raise\').astype(int)\n        require(set(frame.sentence_id) == allowed and not frame.sentence_id.duplicated().any(), f\'{path.name}: meaning set differs from baseline450\')\n        require(frame.tag.eq(frame.text).all(), f\'{path.name}: tag/text mismatch; intervention rows are forbidden\')\n        require(frame.source.nunique() == frame.tag.nunique() == 1, f\'{path.name}: mixed directions\')\n        pair = str(frame.source.iloc[0]) + \'__\' + str(frame.tag.iloc[0])\n        require(not frame.source.eq(frame.tag).any(), f\'{path.name}: self direction\')\n        require(path.stem == pair + \'__\' + str(frame.tag.iloc[0]), f\'{path.name}: filename/direction mismatch\')\n        components = [\'source_tag\', \'target_tag\', \'eos\', \'unknown\', \'punctuation\', \'lexical\'] if study == \'it2\' else [\'source_code\', \'eos\', \'punctuation\', \'lexical\']\n        require(set(components) <= set(frame), f\'{path.name}: component columns differ from the original model protocol\')\n        mass = frame[components].apply(pd.to_numeric, errors=\'raise\').to_numpy(float)\n        require(np.isfinite(mass).all() and (mass >= -1e-7).all() and (mass <= 1+1e-5).all(), f\'{path.name}: invalid component masses\')\n        require(np.allclose(mass.sum(1), 1, atol=1e-5, rtol=0), f\'{path.name}: component closure failed\')\n        require(np.isfinite(frame.lm).all() and frame.lm.between(0, 1).all(), f\'{path.name}: invalid lexical mass\')\n        require(np.allclose(frame.lm, frame.lexical, atol=1e-7, rtol=0), f\'{path.name}: LM differs from lexical share\')\n        require((frame.closure_maxerr < 1e-5).all() and (frame.source_len > 0).all() and (frame.target_len > 0).all(), f\'{path.name}: recorded closure/length gate failed\')\n        frame[\'pair\'] = pair\n        rows.append(frame[[\'sentence_id\', \'corpus_sentence_id\', \'pair\', \'lm\']])\n        if i == 1 or i % 5 == 0 or i == len(files):\n            progress(f\'{study}: validated CSV {i}/40; {time.perf_counter()-started:.1f}s\')\n    frame = pd.concat(rows, ignore_index=True)\n    require(len(frame) == 18000 and not frame.duplicated([\'sentence_id\', \'pair\']).any(), f\'{study}: incomplete/duplicate panel\')\n    require(frame.groupby(\'sentence_id\').pair.nunique().eq(40).all(), f\'{study}: an incomplete meaning block\')\n    require(frame.groupby(\'sentence_id\').corpus_sentence_id.nunique().eq(1).all(), f\'{study}: corpus identities disagree across directions\')\n    manifest = pd.read_csv(Path(root)/\'sample_identity.csv\', usecols=[\'pair\', \'sentence_id\', \'corpus_sentence_id\'], dtype={\'corpus_sentence_id\': str})\n    reference = manifest[manifest.sentence_id.isin(allowed)]\n    joined = frame.merge(reference, on=[\'pair\', \'sentence_id\'], how=\'left\', validate=\'one_to_one\', suffixes=(\'\', \'_manifest\'))\n    require(joined.corpus_sentence_id.eq(joined.corpus_sentence_id_manifest).all(), f\'{study}: corpus IDs disagree with the hash-verified original manifest\')\n    pairs = sorted(frame.pair.unique()); ids = sorted(allowed)\n    languages = sorted({l for p in pairs for l in p.split(\'__\')})\n    require(len(languages) == 11, f\'{study}: expected 11 languages\')\n    pivot = frame.pivot(index=\'sentence_id\', columns=\'pair\', values=\'lm\').reindex(index=ids, columns=pairs)\n    Y = pivot.to_numpy(float)\n    require(Y.shape == (450, 40) and np.isfinite(Y).all(), f\'{study}: invalid matrix\')\n    identity = frame[[\'sentence_id\', \'corpus_sentence_id\']].drop_duplicates().set_index(\'sentence_id\').loc[ids].corpus_sentence_id.astype(str).tolist()\n    return {\'study\': study, \'Y\': Y, \'ids\': ids, \'pairs\': pairs, \'languages\': languages,\n            \'corpus_ids\': identity, \'csv_sha256\': hashes, \'run_lock_sha256\': sha256_file(lock_path)}\n\ndef validate_root(root, expected, original_split_sha256):\n    root = Path(root)\n    for name, digest in [(\'frozen_meaning_split.json\', original_split_sha256), (\'sample_identity.csv\', expected[\'sample_sha256\'])]:\n        path = root / name\n        require(path.is_file(), f\'Missing original input manifest: {path}\')\n        require(sha256_file(path) == digest, f\'Original manifest hash differs: {path}\')\n    split = json.loads((root/\'frozen_meaning_split.json\').read_text(encoding=\'utf-8\'))\n    require(len(split[\'baseline450\']) == 450 and len(split[\'intervention300\']) == 300, \'Original 450/300 split is malformed\')\n    require(not set(split[\'baseline450\']) & set(split[\'intervention300\']), \'Original baseline and intervention overlap\')\n    return split\n\n# E1_SECTION: DESIGN\ndef reverse(pair):\n    a, b = pair.split(\'__\'); return b + \'__\' + a\n\ndef design(pairs, languages, method):\n    n = len(languages)\n    if method == \'direction_mean\':\n        raise ValueError(\'Direction mean is represented by identity; unseen-dyad prediction is not estimable\')\n    blocks = [np.ones((len(pairs), 1))]\n    for side, include in [(0, method in (\'source\', \'both\')), (1, method in (\'target\', \'both\'))]:\n        if include:\n            block = np.full((len(pairs), n), -1/n)\n            for i, p in enumerate(pairs): block[i, languages.index(p.split(\'__\')[side])] += 1\n            blocks.append(block)\n    return np.column_stack(blocks)\n\ndef graph_connected(pairs, languages):\n    adjacency = {l: set() for l in languages}\n    for p in pairs:\n        a, b = p.split(\'__\'); adjacency[a].add(b); adjacency[b].add(a)\n    seen = {languages[0]}; stack = [languages[0]]\n    while stack:\n        for neighbor in adjacency[stack.pop()] - seen:\n            seen.add(neighbor); stack.append(neighbor)\n    return len(seen) == len(languages)\n\ndef build_operators(pairs, languages):\n    """Prediction matrices exactly refit OLS to each resampled training panel.\n\n    Meaning resampling preserves the same count for every direction, so OLS on\n    pooled rows equals OLS on resampled direction means. Design/pseudoinverses\n    are fixed by the graph; coefficients/predictions change on every draw.\n    """\n    require(len(pairs) == 40 and len(set(pairs)) == 40, \'Expected 40 unique directions\')\n    require(all(reverse(p) in pairs for p in pairs), \'Missing reciprocal direction\')\n    dyads = [p for p in pairs if p < reverse(p)]\n    require(len(dyads) == 20, \'Expected 20 dyads\')\n    full = {}; held = {}; audits = []\n    expected_rank = {\'mean\': 1, \'source\': len(languages), \'target\': len(languages), \'both\': 2*len(languages)-1}\n    for method in MODEL_ORDER:\n        if method == \'direction_mean\':\n            full[method] = np.eye(len(pairs)); continue\n        X = design(pairs, languages, method)\n        require(np.linalg.matrix_rank(X) == expected_rank[method], f\'Full {method} design not identifiable\')\n        full[method] = X @ np.linalg.pinv(X)\n        H = np.zeros((len(pairs), len(pairs)))\n        for dyad in dyads:\n            test = [pairs.index(dyad), pairs.index(reverse(dyad))]\n            train = [i for i in range(len(pairs)) if i not in test]\n            connected = graph_connected([pairs[i] for i in train], languages)\n            rank = int(np.linalg.matrix_rank(X[train]))\n            audits.append({\'dyad\': dyad, \'method\': method, \'connected\': connected,\n                           \'training_directions\': len(train), \'rank\': rank, \'expected_rank\': expected_rank[method],\n                           \'identifiable\': rank == expected_rank[method]})\n            require(connected and rank == expected_rank[method], f\'Dyad {dyad}: {method} fold invalid; rank={rank}, connected={connected}\')\n            H[np.ix_(test, train)] = X[test] @ np.linalg.pinv(X[train])\n            require(np.all(H[np.ix_(test, test)] == 0), \'Held-dyad labels leaked into prediction operator\')\n        held[method] = H\n    return {\'full\': full, \'held\': held, \'dyads\': dyads}, pd.DataFrame(audits)\n\ndef split_indices(panel, split):\n    lookup = {v: i for i, v in enumerate(panel[\'ids\'])}\n    require(set(split[\'train\']) | set(split[\'validation\']) | set(split[\'test\']) == set(panel[\'ids\']), \'E1 split does not cover exactly baseline450\')\n    pieces = [set(split[k]) for k in (\'train\', \'validation\', \'test\')]\n    require(not pieces[0]&pieces[1] and not pieces[0]&pieces[2] and not pieces[1]&pieces[2], \'E1 meaning leakage\')\n    require([len(x) for x in pieces] == [315, 67, 68], \'E1 split sizes changed\')\n    return {k: np.array([lookup[i] for i in split[k]], int) for k in (\'train\', \'validation\', \'test\')}\n\ndef fit_coefficients(panel, train_Y, operators):\n    rows = []; mean = train_Y.mean(0); pairs = panel[\'pairs\']; langs = panel[\'languages\']; n = len(langs)\n    specifications = [(\'all_directions\', None, list(range(40)))]\n    specifications += [(\'held_dyad\', p, [i for i, q in enumerate(pairs) if q not in {p, reverse(p)}]) for p in operators[\'dyads\']]\n    for protocol, dyad, keep in specifications:\n        for method in (\'mean\', \'source\', \'target\', \'both\'):\n            coef = np.linalg.pinv(design([pairs[i] for i in keep], langs, method)) @ mean[keep]\n            rows.append({\'study\': panel[\'study\'], \'fit\': protocol, \'held_dyad\': dyad, \'method\': method,\n                         \'term\': \'intercept\', \'language\': \'\', \'coefficient\': float(coef[0])})\n            at = 1\n            for role, enabled in [(\'source\', method in (\'source\',\'both\')), (\'target\', method in (\'target\',\'both\'))]:\n                if enabled:\n                    for language, value in zip(langs, coef[at:at+n]):\n                        rows.append({\'study\': panel[\'study\'], \'fit\': protocol, \'held_dyad\': dyad, \'method\': method,\n                                     \'term\': role, \'language\': language, \'coefficient\': float(value)})\n                    at += n\n    return pd.DataFrame(rows)\n\n# E1_SECTION: SCORES\ndef safe_ratio_r2(sse, denominator):\n    return np.divide(sse, denominator, out=np.full(np.shape(sse), np.nan, float), where=denominator > 1e-15) * -1 + 1\n\ndef batch_scores(prediction, Y_eval, weights, reference_prediction):\n    """Metrics over complete meaning blocks; predictions are constant per direction.\n\n    R2_test_centered uses the current evaluation grand mean. The training-mean\n    reference uses the fitted mean-only predictor, excluding the held dyad in\n    every LODO fold. Both reference definitions are reported explicitly.\n    """\n    P = np.asarray(prediction, float); W = np.asarray(weights, float)\n    size = W.sum(1, keepdims=True); require((size > 0).all(), \'Empty bootstrap evaluation block\')\n    mu = W @ Y_eval / size; second = W @ (Y_eval**2) / size\n    row_sse = np.maximum(np.mean(second - 2*P*mu + P**2, axis=1), 0)\n    row_sst = np.maximum(second.mean(1) - mu.mean(1)**2, 0)\n    row_reference_sse = np.maximum(np.mean(second - 2*reference_prediction*mu + reference_prediction**2, axis=1), 0)\n    absolute = np.abs(Y_eval[None, :, :] - P[:, None, :])\n    row_mae = np.einsum(\'btd,bt->b\', absolute, W, optimize=True) / (size[:, 0]*Y_eval.shape[1])\n    direction_sse = np.mean((mu-P)**2, axis=1)\n    direction_sst = np.mean((mu-mu.mean(1, keepdims=True))**2, axis=1)\n    direction_reference_sse = np.mean((mu-reference_prediction)**2, axis=1)\n    return np.stack([\n        np.column_stack([safe_ratio_r2(row_sse,row_sst), safe_ratio_r2(row_sse,row_reference_sse), row_mae, np.sqrt(row_sse)]),\n        np.column_stack([safe_ratio_r2(direction_sse,direction_sst), safe_ratio_r2(direction_sse,direction_reference_sse), np.mean(np.abs(mu-P),axis=1), np.sqrt(direction_sse)])\n    ], axis=1)\n\ndef protocol_data(train_Y, test_Y, operators):\n    return {\n        \'unseen_meanings\': (test_Y, operators[\'full\']),\n        \'unseen_dyad_shared_meanings\': (train_Y, operators[\'held\']),\n        \'unseen_meanings_and_dyad\': (test_Y, operators[\'held\'])}\n\ndef point_evaluation(panel, split, operators, evaluation_partition=\'test\', predictions=True):\n    indices = split_indices(panel, split); train_Y = panel[\'Y\'][indices[\'train\']]\n    eval_Y = panel[\'Y\'][indices[evaluation_partition]]\n    mean_train = train_Y.mean(0); summaries=[]; perdyad=[]; frames=[]\n    for protocol, (Y, Hs) in protocol_data(train_Y, eval_Y, operators).items():\n        if evaluation_partition == \'validation\' and protocol == \'unseen_dyad_shared_meanings\': continue\n        identity_indices = indices[\'train\'] if protocol == \'unseen_dyad_shared_meanings\' else indices[evaluation_partition]\n        reference = (Hs[\'mean\'] @ mean_train)[None, :]\n        prediction_table = {\'study\': panel[\'study\'], \'protocol\': protocol,\n            \'sentence_id\': np.repeat(np.array(panel[\'ids\'])[identity_indices], 40),\n            \'corpus_sentence_id\': np.repeat(np.array(panel[\'corpus_ids\'])[identity_indices], 40),\n            \'pair\': np.tile(panel[\'pairs\'], len(Y)), \'observed_lm\': Y.reshape(-1)}\n        for method in MODEL_ORDER:\n            if method not in Hs:\n                for scale in SCALES:\n                    summaries.append({\'study\':panel[\'study\'],\'partition\':evaluation_partition,\'protocol\':protocol,\'method\':method,\'scale\':scale,\n                        \'available\':False,\'reason\':\'Unrestricted direction mean cannot estimate an unseen dyad without an extra prediction rule.\'})\n                continue\n            P = Hs[method] @ mean_train\n            scores = batch_scores(P[None,:], Y, np.ones((1,len(Y))), reference)[0]\n            for si,scale in enumerate(SCALES):\n                summaries.append({\'study\':panel[\'study\'],\'partition\':evaluation_partition,\'protocol\':protocol,\'method\':method,\'scale\':scale,\n                    \'available\':True,\'reason\':\'\',\'evaluation_meanings\':len(Y),\'evaluation_rows\':int(Y.size),\n                    **{metric:float(scores[si,mi]) for mi,metric in enumerate(METRICS)}})\n            prediction_table[\'pred_\'+method] = np.tile(P, len(Y))\n            for dyad in operators[\'dyads\']:\n                keep = [panel[\'pairs\'].index(dyad), panel[\'pairs\'].index(reverse(dyad))]\n                error = Y[:,keep] - P[keep]; direction_error = Y[:,keep].mean(0)-P[keep]\n                for scale,err in [(\'individual_rows\',error),(\'direction_means\',direction_error)]:\n                    perdyad.append({\'study\':panel[\'study\'],\'partition\':evaluation_partition,\'protocol\':protocol,\'dyad\':dyad,\n                        \'method\':method,\'scale\':scale,\'evaluation_meanings\':len(Y),\'MAE\':float(np.abs(err).mean()),\'RMSE\':float(np.sqrt(np.mean(err**2)))})\n        if predictions:frames.append(pd.DataFrame(prediction_table))\n    return pd.DataFrame(summaries),pd.DataFrame(perdyad),pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()\n\n# E1_SECTION: BOOTSTRAP\ndef bootstrap_evaluation(panel, split, operators, n_boot, seed, batch_size, checkpoint, fingerprint):\n    idx = split_indices(panel,split); train_Y=panel[\'Y\'][idx[\'train\']]; test_Y=panel[\'Y\'][idx[\'test\']]\n    specs=protocol_data(train_Y,test_Y,operators)\n    keys=[(protocol,method) for protocol,(_,Hs) in specs.items() for method in MODEL_ORDER if method in Hs]\n    samples=np.full((n_boot,len(keys),len(SCALES),len(METRICS)),np.nan)\n    rng=np.random.default_rng(seed); completed=0; draw_chain=\'\'\n    if Path(checkpoint).is_file():\n        with np.load(checkpoint,allow_pickle=False) as saved:\n            require(str(saved[\'fingerprint\'].item())==fingerprint,\'Bootstrap checkpoint settings/input hash changed\')\n            require(saved[\'samples\'].shape==samples.shape,\'Bootstrap checkpoint shape changed\')\n            samples[:]=saved[\'samples\'];completed=int(saved[\'completed\'].item())\n            rng.bit_generator.state=json.loads(str(saved[\'rng_state\'].item()))\n            draw_chain=str(saved[\'draw_chain\'].item())\n        progress(f\'{panel["study"]}: resumed bootstrap {completed:,}/{n_boot:,}\')\n    started=time.perf_counter()\n    for start in range(completed,n_boot,batch_size):\n        stop=min(start+batch_size,n_boot);b=stop-start\n        Wtrain=rng.multinomial(len(train_Y),np.full(len(train_Y),1/len(train_Y)),size=b)\n        Wtest=rng.multinomial(len(test_Y),np.full(len(test_Y),1/len(test_Y)),size=b)\n        draw_chain=sha256_bytes(draw_chain.encode()+Wtrain.tobytes()+Wtest.tobytes())\n        train_mean=Wtrain@train_Y/len(train_Y)\n        for ki,(protocol,method) in enumerate(keys):\n            Y,Hs=specs[protocol];weights=Wtrain if protocol==\'unseen_dyad_shared_meanings\' else Wtest\n            P=train_mean@Hs[method].T;reference=train_mean@Hs[\'mean\'].T\n            samples[start:stop,ki]=batch_scores(P,Y,weights,reference)\n        if start==completed or stop==n_boot or stop//1000>start//1000:\n            atomic_npz(checkpoint,samples=samples,completed=np.array(stop),fingerprint=np.array(fingerprint),\n                rng_state=np.array(json.dumps(rng.bit_generator.state)),draw_chain=np.array(draw_chain),\n                keys=np.array([\'|\'.join(k) for k in keys]),scales=np.array(SCALES),metrics=np.array(METRICS))\n            progress(f\'{panel["study"]}: bootstrap {stop:,}/{n_boot:,}; {time.perf_counter()-started:.1f}s this session\')\n    return samples,keys,draw_chain\n\ndef add_intervals(point,samples,keys):\n    result=point.copy()\n    for i,row in result.iterrows():\n        if not row.available:continue\n        ki=keys.index((row.protocol,row.method));si=SCALES.index(row.scale)\n        for mi,metric in enumerate(METRICS):\n            values=samples[:,ki,si,mi];finite=values[np.isfinite(values)]\n            lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)\n            result.loc[i,metric+\'_ci_low\']=lo;result.loc[i,metric+\'_ci_high\']=hi\n            result.loc[i,metric+\'_bootstrap_valid\']=len(finite)\n    return result\n\ndef paired_contrasts(point,samples,keys):\n    rows=[]\n    for protocol in PROTOCOLS:\n        both=keys.index((protocol,\'both\'));target=keys.index((protocol,\'target\'))\n        for si,scale in enumerate(SCALES):\n            b=point[(point.protocol==protocol)&(point.method==\'both\')&(point.scale==scale)].iloc[0]\n            t=point[(point.protocol==protocol)&(point.method==\'target\')&(point.scale==scale)].iloc[0]\n            for mi,metric in enumerate(METRICS):\n                sign=1 if metric.startswith(\'R2\') else -1\n                values=sign*(samples[:,both,si,mi]-samples[:,target,si,mi]);finite=values[np.isfinite(values)]\n                lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)\n                rows.append({\'study\':b.study,\'protocol\':protocol,\'scale\':scale,\'contrast\':\'both_vs_target\',\n                    \'metric\':metric,\'positive_means\':\'higher R2\' if sign==1 else \'lower prediction error\',\n                    \'estimate\':sign*(b[metric]-t[metric]),\'ci_low\':lo,\'ci_high\':hi,\'bootstrap_valid\':len(finite)})\n    return pd.DataFrame(rows)\n\n# E1_SECTION: RUN\ndef run_study(panel, split, operators, output, n_boot, seed, batch_size, code_hash):\n    output=Path(output);output.mkdir(parents=True,exist_ok=True)\n    signature={\'study\':panel[\'study\'],\'input_csv_sha256\':panel[\'csv_sha256\'],\n        \'split\':split,\'n_boot\':n_boot,\'seed\':seed,\'batch_size\':batch_size,\'code_sha256\':code_hash}\n    fingerprint=sha256_bytes(json.dumps(signature,sort_keys=True).encode())\n    freeze_json(output/\'analysis_lock.json\',signature)\n    done=output/\'completed.json\'\n    if done.exists():\n        stamp=json.loads(done.read_text())\n        require(stamp[\'fingerprint\']==fingerprint,\'Existing completed result has a different signature\')\n        require(all((output/name).is_file() and sha256_file(output/name)==digest for name,digest in stamp[\'result_sha256\'].items()),\'A completed output is missing or modified\')\n        progress(f\'{panel["study"]}: completed results verified; reusing saved analysis\')\n        return pd.read_csv(output/\'test_metrics_with_CI.csv\'),pd.read_csv(output/\'paired_model_contrasts.csv\')\n    idx=split_indices(panel,split)\n    atomic_csv(output/\'training_coefficients.csv\',fit_coefficients(panel,panel[\'Y\'][idx[\'train\']],operators))\n    validation,validation_dyads,_=point_evaluation(panel,split,operators,\'validation\',False)\n    atomic_csv(output/\'validation_point_metrics.csv\',validation)\n    atomic_csv(output/\'validation_per_dyad.csv\',validation_dyads)\n    point,dyads,predictions=point_evaluation(panel,split,operators,\'test\',True)\n    atomic_csv(output/\'test_point_metrics.csv\',point);atomic_csv(output/\'test_per_dyad.csv\',dyads)\n    atomic_csv(output/\'held_out_row_predictions.csv\',predictions)\n    samples,keys,draw_chain=bootstrap_evaluation(panel,split,operators,n_boot,seed,batch_size,output/\'bootstrap_checkpoint.npz\',fingerprint)\n    summary=add_intervals(point,samples,keys);contrasts=paired_contrasts(point,samples,keys)\n    atomic_csv(output/\'test_metrics_with_CI.csv\',summary);atomic_csv(output/\'paired_model_contrasts.csv\',contrasts)\n    names=[\'training_coefficients.csv\',\'validation_point_metrics.csv\',\'validation_per_dyad.csv\',\n        \'test_point_metrics.csv\',\'test_per_dyad.csv\',\'held_out_row_predictions.csv\',\n        \'test_metrics_with_CI.csv\',\'paired_model_contrasts.csv\',\'bootstrap_checkpoint.npz\']\n    atomic_json(done,{\'fingerprint\':fingerprint,\'bootstrap_draw_chain\':draw_chain,\'bootstrap_completed\':n_boot,\n        \'result_sha256\':{name:sha256_file(output/name) for name in names}})\n    progress(f\'{panel["study"]}: E1 finished; saved to {output}\')\n    return summary,contrasts\n'
WORKER_SOURCE='"""Pinned fp32 MADLAD TF collector. No MT generation and no historical tensor reads."""\nimport argparse\nimport gc\nimport importlib.metadata\nimport json\nimport os\nimport time\nimport unicodedata\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom e1_core import progress, require, sha256_file, sha256_bytes, atomic_csv, atomic_json, atomic_npz, freeze_json\n\nCOMPONENTS=[\'target_tag\',\'eos\',\'unknown\',\'punctuation\',\'lexical\']\n\ndef punctuation(piece):\n    text=str(piece).replace(\'▁\',\'\').replace(\'Ġ\',\'\').strip()\n    return not text or all(unicodedata.category(ch)[0] in {\'P\',\'S\',\'Z\',\'C\'} for ch in text)\n\ndef positions(tok, ids, mask, tag_id=None):\n    active=[i for i,v in enumerate(mask) if int(v)]\n    require(active and ids[active[-1]]==tok.eos_token_id,\'Last attended source/target position is not EOS\')\n    pieces=tok.convert_ids_to_tokens(ids)\n    cats={k:[] for k in COMPONENTS}\n    cats[\'eos\']=[active[-1]]\n    body=active[:-1]\n    if tag_id is not None:\n        cats[\'target_tag\']=[i for i in body if ids[i]==tag_id]\n        require(len(cats[\'target_tag\'])==1,\'Expected exactly one target-language control token\')\n        # A standalone SentencePiece separator may precede this token.\n        require(all(not pieces[i].replace(\'▁\',\'\').strip() for i in body if i<cats[\'target_tag\'][0]),\n                \'Unexpected content before the language control token\')\n        body=[i for i in body if i not in cats[\'target_tag\']]\n    cats[\'unknown\']=[i for i in body if ids[i]==tok.unk_token_id]\n    body=[i for i in body if i not in cats[\'unknown\']]\n    require(not any(ids[i] in set(tok.all_special_ids) for i in body),\'Unexpected special token inside text\')\n    # Language codes other than the intentional encoder prefix must not occur as body tokens.\n    require(not any(str(pieces[i]).startswith(\'<2\') and str(pieces[i]).endswith(\'>\') for i in body),\n            \'Embedded language-control token in text\')\n    cats[\'punctuation\']=[i for i in body if punctuation(pieces[i])]\n    cats[\'lexical\']=[i for i in body if i not in cats[\'punctuation\']]\n    require(sorted(sum(cats.values(),[]))==active,\'Position partition does not cover valid positions exactly once\')\n    return cats\n\ndef load_texts(config):\n    for name in config[\'text_candidates\']:\n        p=Path(name)\n        if p.is_file():\n            wide=pd.read_csv(p,dtype={\'id\':str},keep_default_na=False)\n            if \'id\' in wide and set(config[\'languages\'])<=set(wide):\n                require(not wide.id.duplicated().any(),\'Duplicate corpus identities in text cache\')\n                progress(f\'Using original source/reference cache: {p}\')\n                return wide.set_index(\'id\')\n    progress(\'Recovering the original pinned FLORES+ corpus (HF_TOKEN with accepted access required)...\')\n    from datasets import load_dataset\n    token=os.environ.get(\'HF_TOKEN\')\n    require(token,\'Original gated corpus unavailable: enable HF_TOKEN or provide the previous text cache\')\n    ds=load_dataset(config[\'dataset\'],revision=config[\'dataset_revision\'],split=\'devtest\',token=token)\n    long=ds.to_pandas();long[\'language\']=long.iso_639_3.astype(str)+\'_\'+long.iso_15924.astype(str)\n    long=long[long.language.isin(config[\'languages\'])]\n    require(not long.duplicated([\'id\',\'language\']).any(),\'Duplicate corpus rows\')\n    wide=long.pivot(index=\'id\',columns=\'language\',values=\'text\').reset_index()\n    wide[\'id\']=wide.id.astype(str)\n    require(not wide.isna().any().any(),\'Missing original language texts\')\n    atomic_csv(config[\'text_candidates\'][0],wide)\n    return wide.set_index(\'id\')\n\ndef prepare_records(manifest,wide):\n    records=[]\n    for row in manifest.itertuples(index=False):\n        source,target=row.pair.split(\'__\');corpus=str(row.corpus_sentence_id)\n        require(corpus in wide.index,\'Original meaning is absent from text cache\')\n        s=str(wide.loc[corpus,source]);t=str(wide.loc[corpus,target])\n        require(sha256_bytes(s.encode())==row.source_sha256 and sha256_bytes(t.encode())==row.target_sha256,\n                \'Source/reference hashes disagree with original sample; substitution forbidden\')\n        records.append({\'sentence_id\':int(row.sentence_id),\'corpus_sentence_id\':corpus,\'pair\':row.pair,\n                        \'source\':source,\'target\':target,\'src_text\':s,\'tgt_text\':t})\n    return records\n\ndef run(config):\n    require(sha256_file(__file__)==config[\'worker_sha256\'],\'Collector differs from the frozen notebook\')\n    progress(\'Importing pinned PyTorch/Transformers...\')\n    import torch\n    from transformers import T5Tokenizer,T5ForConditionalGeneration\n    from huggingface_hub import snapshot_download\n    require(torch.cuda.is_available(),\'Select an L4 GPU runtime; no impractical CPU inference fallback\')\n    torch.manual_seed(config[\'seed\']);np.random.seed(config[\'seed\'])\n    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False\n    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True\n    torch.set_float32_matmul_precision(\'highest\')\n    output=Path(config[\'output\']);output.mkdir(parents=True,exist_ok=True)\n    manifest=pd.read_csv(config[\'manifest\'],dtype={\'corpus_sentence_id\':str})\n    require(sha256_file(config[\'manifest\'])==config[\'manifest_sha256\'],\'Staged input manifest changed\')\n    require(len(manifest)==18000 and not manifest.duplicated([\'pair\',\'sentence_id\']).any(),\'Expected complete 450 by 40 input\')\n    records=prepare_records(manifest,load_texts(config));progress(\'All 18,000 original source/reference identities verified\')\n    spec=config[\'model\']\n    progress(\'Downloading/verifying pinned MADLAD assets (~11.8 GB weights on first run)...\')\n    directory=snapshot_download(spec[\'repo\'],revision=spec[\'revision\'],local_dir=config[\'model_cache\'],\n        allow_patterns=list(spec[\'files\']))\n    for name,expected in spec[\'files\'].items():\n        require(sha256_file(Path(directory)/name)==expected,f\'MADLAD asset checksum mismatch: {name}\')\n        progress(f\'Verified model asset: {name}\')\n    tok=T5Tokenizer.from_pretrained(directory,local_files_only=True,legacy=False)\n    require(tok.pad_token_id==1 and tok.eos_token_id==2 and tok.unk_token_id==0,\'Unexpected MADLAD token IDs\')\n    coverage=[]\n    for language,code in config[\'language_codes\'].items():\n        piece=\'<2\'+code+\'>\';idx=tok.convert_tokens_to_ids(piece)\n        require(idx==config[\'expected_tag_ids\'][language] and idx!=tok.unk_token_id,\'Language-control ID changed\')\n        require(tok(piece,add_special_tokens=False).input_ids.count(idx)==1,\'Language tag does not tokenize atomically\')\n        coverage.append({\'language\':language,\'control_token\':piece,\'token_id\':idx,\'coverage\':\'published model metadata + real tokenizer\'})\n    atomic_csv(output/\'language_coverage.csv\',pd.DataFrame(coverage))\n    progress(\'Loading fp32 MADLAD on the GPU. No quantization, autocast or CPU offloading...\')\n    model=T5ForConditionalGeneration.from_pretrained(directory,local_files_only=True,use_safetensors=True,\n        torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation=\'eager\',device_map={\'\':\'cuda:0\'})\n    model.eval();require(all(p.dtype==torch.float32 and p.device.type==\'cuda\' for p in model.parameters()),\'Model is not wholly fp32/CUDA\')\n    require(model.config.decoder_start_token_id==0,\'MADLAD decoder start ID differs from pinned config\')\n    signature={\'model\':spec,\'seed\':config[\'seed\'],\'worker_sha256\':config[\'worker_sha256\'],\n        \'manifest_sha256\':config[\'manifest_sha256\'],\'precision\':\'float32\',\'tf32\':False,\'attention\':\'eager\',\n        \'num_parameters\':sum(p.numel() for p in model.parameters()),\'decoder_start_token_id\':0,\n        \'layers\':model.config.num_decoder_layers,\'heads\':model.config.num_heads,\n        \'gpu\':torch.cuda.get_device_name(0),\n        \'packages\':{p:importlib.metadata.version(p) for p in [\'torch\',\'transformers\',\'tokenizers\',\'sentencepiece\',\'numpy\',\'pandas\',\'datasets\',\'accelerate\']},\n        \'target_rows\':\'lexical reference labels: exclude EOS, punctuation, unknown; decoder inputs shifted by model\',\n        \'source_unknown\':\'separate component; also save LM including unknown for sensitivity\',\n        \'truncation\':False,\'batch_size\':1,\'max_supported_token_length\':config[\'max_tokens\']}\n    freeze_json(output/\'run_lock.json\',signature)\n    fingerprint=sha256_bytes(json.dumps(signature,sort_keys=True).encode())\n\n    def encode(record,audit=False):\n        x=tok(\'<2\'+config[\'language_codes\'][record[\'target\']]+\'> \'+record[\'src_text\'],return_tensors=\'pt\',truncation=False)\n        y=tok(record[\'tgt_text\'],return_tensors=\'pt\',truncation=False)\n        require(max(x.input_ids.shape[1],y.input_ids.shape[1])<=config[\'max_tokens\'],\n                \'Sequence exceeds the safety ceiling; no truncation/drop was applied. Inspect length and raise ceiling in a new run directory if GPU permits.\')\n        sid=x.input_ids[0].tolist();tid=y.input_ids[0].tolist()\n        cats=positions(tok,sid,x.attention_mask[0].tolist(),config[\'expected_tag_ids\'][record[\'target\']])\n        tgt=positions(tok,tid,y.attention_mask[0].tolist());rows=tgt[\'lexical\']\n        require(cats[\'lexical\'] and rows,\'No lexical source/target positions; stop rather than silently drop a meaning\')\n        if audit:\n            a={\'pair\':record[\'pair\'],\'sentence_id\':record[\'sentence_id\'],\'source_tokens\':list(enumerate(tok.convert_ids_to_tokens(sid))),\n               \'source_attention_mask\':x.attention_mask[0].tolist(),\'source_positions\':cats,\n               \'reference_tokens\':list(enumerate(tok.convert_ids_to_tokens(tid))),\'target_lexical_label_positions\':rows,\n               \'decoder_input_ids\':model.prepare_decoder_input_ids_from_labels(y.input_ids)[0].tolist(),\n               \'note\':\'Decoder start ID 0 is <unk>, not pad ID 1. Attention rows are selected by reference label position.\'}\n            print(json.dumps(a,ensure_ascii=False),flush=True)\n        return x,y,cats,rows\n\n    def collect_one(record,audit=False):\n        x,y,cats,rows=encode(record,audit)\n        with torch.inference_mode():\n            result=model(input_ids=x.input_ids.cuda(),attention_mask=x.attention_mask.cuda(),labels=y.input_ids.cuda(),\n                         output_attentions=True,use_cache=False,return_dict=True)\n            require(len(result.cross_attentions)==model.config.num_decoder_layers,\'Cross-attention unavailable\')\n            layer_shares=[];closure=0.0\n            for attention in result.cross_attentions:\n                require(attention.dtype==torch.float32,\'Non-fp32 attention\')\n                a=attention[0].index_select(1,torch.tensor(rows,device=attention.device))\n                require(bool(torch.isfinite(a).all()) and float(a.min())>=-1e-7,\'Nonfinite/negative attention\')\n                closure=max(closure,float((a.sum(-1)-1).abs().max()))\n                h=torch.stack([a[:,:,cats[k]].sum(-1).mean(-1) for k in COMPONENTS],dim=-1)\n                layer_shares.append(h.cpu().numpy())\n            heads=np.stack(layer_shares).astype(np.float32)\n            require(closure<1e-5 and np.max(np.abs(heads.sum(-1)-1))<1e-5,\'Per-head/target-row attention closure failed\')\n            require(np.isfinite(heads).all(),\'Invalid head component shares\')\n            means=heads.mean((0,1));nll=float(result.loss)\n        row={k:record[k] for k in [\'sentence_id\',\'corpus_sentence_id\',\'pair\',\'source\',\'target\']}\n        row.update({\'eligible\':True,\'reason\':\'\',\'lm\':float(means[-1]),\'nll\':nll,\'source_len\':len(cats[\'lexical\']),\n            \'target_len\':len(rows),\'source_total_tokens\':int(x.input_ids.shape[1]),\'target_total_tokens\':int(y.input_ids.shape[1]),\n            \'source_unknown_tokens\':len(cats[\'unknown\']),\'closure_maxerr\':closure,\n            **{k:float(v) for k,v in zip(COMPONENTS,means)}})\n        row[\'lm_with_unknown\']=row[\'lexical\']+row[\'unknown\']\n        row[\'conditional_lexical\']=row[\'lexical\']/(row[\'lexical\']+row[\'punctuation\'])\n        del result,layer_shares,x,y\n        return row,heads\n\n    def collect(record,audit=False):\n        try:\n            return collect_one(record,audit)\n        except Exception as exc:\n            atomic_json(output/\'last_failure.json\',{\'pair\':record[\'pair\'],\'sentence_id\':record[\'sentence_id\'],\n                \'corpus_sentence_id\':record[\'corpus_sentence_id\'],\'error_type\':type(exc).__name__,\n                \'error\':str(exc),\'action\':\'Stopped without exclusion, truncation or precision fallback. Completed chunks remain reusable.\'})\n            raise\n\n    # All language/direction supports are checked before committing to a long run.\n    progress(\'Pre-tokenizing all 18,000 cells to verify positions, lexical support and absence of truncation...\')\n    for i,record in enumerate(records,1):\n        encode(record)\n        if i%2000==0:progress(f\'Token preflight {i:,}/18,000\')\n    chosen=[next(r for r in records if r[\'target\']==lang) for lang in config[\'languages\']]\n    smoke=[];torch.cuda.reset_peak_memory_stats();smoke_start=time.perf_counter()\n    for i,record in enumerate(chosen):\n        row,_=collect(record,audit=i<2);smoke.append(row)\n    torch.cuda.synchronize();seconds=time.perf_counter()-smoke_start\n    original,_=collect(chosen[0]);repeated,_=collect(chosen[0])\n    require(abs(original[\'lm\']-repeated[\'lm\'])<1e-7,\'Same fp32 TF cell does not reproduce\')\n    atomic_json(output/\'smoke_report.json\',{\'passed\':True,\'tested_targets\':config[\'languages\'],\n        \'rows\':len(smoke),\'seconds\':seconds,\'rough_full_hours\':seconds/len(smoke)*18000/3600,\n        \'peak_gpu_GiB\':torch.cuda.max_memory_allocated()/2**30,\'max_closure_error\':max(r[\'closure_maxerr\'] for r in smoke),\n        \'repeated_cell_lm_delta\':abs(original[\'lm\']-repeated[\'lm\'])})\n    progress(f\'Smoke passed: {len(smoke)/seconds:.2f} rows/s; rough full 18,000-row compute {seconds/len(smoke)*18000/3600:.2f} hours (excludes downloads/Drive).\')\n    progress(\'Full TF collection begins; checkpoints every 25 rows. No free generation.\')\n    pairs=sorted({r[\'pair\'] for r in records});completed=[]\n    for pi,pair in enumerate(pairs,1):\n        group=[r for r in records if r[\'pair\']==pair];folder=output/\'parts\'/pair;folder.mkdir(parents=True,exist_ok=True)\n        for start in range(0,len(group),25):\n            block=group[start:start+25];name=f\'{start:04d}\';csv=folder/(name+\'.csv\');npz=folder/(name+\'.npz\');stamp=folder/(name+\'.json\')\n            if stamp.is_file():\n                info=json.loads(stamp.read_text());require(info[\'fingerprint\']==fingerprint,\'Checkpoint run identity differs\')\n                require(sha256_file(csv)==info[\'csv_sha256\'] and sha256_file(npz)==info[\'npz_sha256\'],\'Modified checkpoint\')\n                cached=pd.read_csv(csv)\n                require(cached.sentence_id.tolist()==[r[\'sentence_id\'] for r in block],\'Checkpoint meaning identities differ\')\n                continue\n            tic=time.perf_counter();rows=[];heads=[]\n            for record in block:\n                row,head=collect(record);rows.append(row);heads.append(head)\n            atomic_csv(csv,pd.DataFrame(rows));atomic_npz(npz,shares=np.stack(heads),columns=np.array(COMPONENTS,dtype=\'U16\'),\n                sentence_id=np.array([r[\'sentence_id\'] for r in block],dtype=np.int64))\n            atomic_json(stamp,{\'fingerprint\':fingerprint,\'csv_sha256\':sha256_file(csv),\'npz_sha256\':sha256_file(npz)})\n            progress(f\'MADLAD direction {pi}/40 {pair}: {start+len(block)}/450; block {time.perf_counter()-tic:.1f}s\')\n        frame=pd.concat([pd.read_csv(folder/f\'{i:04d}.csv\',dtype={\'corpus_sentence_id\':str}) for i in range(0,450,25)],ignore_index=True)\n        require(len(frame)==450 and not frame.sentence_id.duplicated().any(),\'Incomplete direction\')\n        dest=output/\'baseline_tf\'/(pair+\'.csv\');atomic_csv(dest,frame);completed.append(frame)\n        gc.collect();torch.cuda.empty_cache()\n    all_rows=pd.concat(completed,ignore_index=True)\n    require(len(all_rows)==18000 and not all_rows.duplicated([\'pair\',\'sentence_id\']).any(),\'Incomplete final panel\')\n    atomic_csv(output/\'all_rows.csv\',all_rows)\n    atomic_json(output/\'complete.json\',{\'fingerprint\':fingerprint,\'worker_sha256\':config[\'worker_sha256\'],\n        \'rows\':18000,\'directions\':40,\'meanings\':450,\'all_rows_sha256\':sha256_file(output/\'all_rows.csv\'),\n        \'run_lock_sha256\':sha256_file(output/\'run_lock.json\'),\n        \'head_summary_format\':\'parts/<pair>/<block>.npz; shape meaning x layer x head x component; Unicode columns and int64 IDs; allow_pickle=False\',\n        \'checkpoint_blocks\':{str(p.relative_to(output)):sha256_file(p) for p in (output/\'parts\').rglob(\'*.json\')}})\n    progress(\'MADLAD fp32 collection complete. All per-row and per-head component shares saved.\')\n\nif __name__==\'__main__\':\n    parser=argparse.ArgumentParser();parser.add_argument(\'--config\',required=True)\n    run(json.loads(Path(parser.parse_args().config).read_text()))\n'
ANALYSIS_SOURCE='"""Three-model E3 summaries on exactly the same complete baseline450 graph."""\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom e1_core import (require, progress, sha256_file, sha256_bytes, atomic_csv, atomic_json,\n                     atomic_npz, freeze_json, design, reverse, build_operators)\n\ndef panel_from_collector(frame,reference,root):\n    require(len(frame)==18000 and not frame.duplicated([\'sentence_id\',\'pair\']).any(),\'MADLAD panel must be complete\')\n    require(frame.eligible.astype(str).str.lower().isin([\'true\',\'1\',\'1.0\']).all(),\'Ineligible MADLAD rows; no silent deletion\')\n    require(set(frame.pair)==set(reference[\'pairs\']) and set(frame.sentence_id)==set(reference[\'ids\']),\'Common graph/meanings changed\')\n    reference_ids=dict(zip(reference[\'ids\'],reference[\'corpus_ids\']))\n    require(frame.corpus_sentence_id.astype(str).eq(frame.sentence_id.map(reference_ids).astype(str)).all(),\'Corpus identity mismatch\')\n    matrix=frame.pivot(index=\'sentence_id\',columns=\'pair\',values=\'lm\').loc[reference[\'ids\'],reference[\'pairs\']].to_numpy(float)\n    require(np.isfinite(matrix).all(),\'Nonfinite MADLAD panel\')\n    root=Path(root)\n    return {\'study\':\'madlad\',\'Y\':matrix,\'ids\':reference[\'ids\'],\'pairs\':reference[\'pairs\'],\'languages\':reference[\'languages\'],\n            \'corpus_ids\':reference[\'corpus_ids\'],\'csv_sha256\':{\'all_rows.csv\':sha256_file(root/\'all_rows.csv\')},\n            \'run_lock_sha256\':sha256_file(root/\'run_lock.json\')}\n\ndef field_design(pairs,languages):\n    canonical=[p for p in pairs if p<reverse(p)]\n    difference=np.zeros((len(pairs),len(canonical)))\n    incidence=np.zeros((len(canonical),len(languages)))\n    for j,p in enumerate(canonical):\n        a,b=p.split(\'__\');difference[pairs.index(p),j]=1;difference[pairs.index(reverse(p)),j]=-1\n        incidence[j,languages.index(a)]=1;incidence[j,languages.index(b)]=-1\n    require(np.linalg.matrix_rank(incidence)==len(languages)-1,\'Conservative field graph disconnected\')\n    return difference,incidence,incidence@np.linalg.pinv(incidence)\n\ndef allocation_statistics(Y,W,P,H,D,G):\n    """Weighted whole-meaning statistics with an exact two-way ANOVA decomposition."""\n    weights=W/W.sum(1,keepdims=True);mu=weights@Y;grand=mu.mean(1)\n    total=weights@(Y*Y).mean(1)-grand*grand\n    direction=np.mean((mu-grand[:,None])**2,axis=1)\n    meaning=weights@(Y.mean(1)**2)-grand*grand\n    residual=total-direction-meaning\n    require(np.all(total>0) and np.all(residual>=-1e-12),\'Invalid row variance partition\')\n    require(np.allclose((direction+meaning+residual)/total,1,atol=1e-8),\'ANOVA energy closure failed\')\n    edge=mu@D;gradient=edge@G\n    energy=np.sum(edge*edge,axis=1)\n    additive=mu@P.T\n    held=mu@H.T\n    # In-sample direction-mean and same-meaning held-dyad metrics are descriptive.\n    return {\'mean_mass\':grand,\'row_direction_share\':direction/total,\'row_meaning_share\':meaning/total,\n        \'row_residual_share\':np.maximum(residual,0)/total,\n        \'additive_direction_R2\':1-np.mean((mu-additive)**2,axis=1)/np.maximum(direction,1e-30),\n        \'held_dyad_direction_R2_shared_meanings\':1-np.mean((mu-held)**2,axis=1)/np.maximum(direction,1e-30),\n        \'gradient_fraction\':np.sum(gradient*gradient,axis=1)/np.maximum(energy,1e-30),\n        \'nonconservative_residual_L2\':np.sqrt(np.sum((edge-gradient)**2,axis=1))}\n\ndef supplementary(panel,components,output,n_boot=10000,seed=20261106):\n    output=Path(output);output.mkdir(parents=True,exist_ok=True)\n    operators,_=build_operators(panel[\'pairs\'],panel[\'languages\'])\n    P=operators[\'full\'][\'both\'];H=operators[\'held\'][\'both\']\n    D,B,G=field_design(panel[\'pairs\'],panel[\'languages\'])\n    cubes={name:components.pivot(index=\'sentence_id\',columns=\'pair\',values=name).loc[panel[\'ids\'],panel[\'pairs\']].to_numpy(float)\n           for name in [\'lexical\',\'eos\',\'language_control\',\'punctuation\',\'unknown\',\'conditional_lexical\',\'lexical_with_unknown\']}\n    require(all(np.isfinite(v).all() for v in cubes.values()),\'Nonfinite component panel\')\n    require(np.allclose(sum(cubes[k] for k in [\'lexical\',\'eos\',\'language_control\',\'punctuation\',\'unknown\']),1,atol=1e-5),\n            \'Aggregated component closure failed\')\n    signature={\'code_sha256\':sha256_file(__file__),\'study\':panel[\'study\'],\'input\':panel[\'csv_sha256\'],\n        \'component_hash\':sha256_bytes(components.sort_values([\'sentence_id\',\'pair\']).to_csv(index=False).encode()),\n        \'n_boot\':n_boot,\'seed\':seed,\'unit\':\'whole meanings jointly across all directions\',\n        \'null\':\'independent random permutation of the 20 observed signed edge values, conditional on fixed graph\'}\n    freeze_json(output/\'allocation_lock.json\',signature)\n    fingerprint=sha256_bytes(json.dumps(signature,sort_keys=True).encode())\n    cache=output/\'allocation_bootstrap.npz\'\n    # The recorded NLLB unknown component is identically zero because its baseline\n    # includes unknown tokens in lexical mass. No zero-variance R² is interpreted.\n    names=[k for k in cubes if k!=\'unknown\']\n    atomic_csv(output/\'component_means.csv\',pd.DataFrame([{\'study\':panel[\'study\'],\'component\':k,\'mean_mass\':float(v.mean())} for k,v in cubes.items()]))\n    point={k:allocation_statistics(cubes[k],np.ones((1,len(panel[\'ids\']))),P,H,D,G) for k in names}\n    metrics=list(point[names[0]])\n    if cache.is_file():\n        with np.load(cache,allow_pickle=False) as f:\n            require(str(f[\'fingerprint\'])==fingerprint,\'Allocation bootstrap cache differs\')\n            samples=f[\'samples\'];null=f[\'null\']\n    else:\n        samples=np.empty((n_boot,len(names),len(metrics)));null=np.empty(n_boot)\n        rng=np.random.default_rng(seed);edge=panel[\'Y\'].mean(0)@D\n        for start in range(0,n_boot,128):\n            count=min(128,n_boot-start);W=rng.multinomial(len(panel[\'ids\']),np.full(len(panel[\'ids\']),1/len(panel[\'ids\'])),size=count)\n            for ci,name in enumerate(names):\n                values=allocation_statistics(cubes[name],W,P,H,D,G)\n                samples[start:start+count,ci,:]=np.column_stack([values[m] for m in metrics])\n            perm=np.stack([rng.permutation(edge) for _ in range(count)])\n            null[start:start+count]=np.sum((perm@G)**2,axis=1)/np.sum(perm*perm,axis=1)\n            if start==0 or (start+count)//1024>start//1024:progress(f\'{panel["study"]}: allocation bootstrap {start+count:,}/{n_boot:,}\')\n        atomic_npz(cache,fingerprint=np.array(fingerprint),samples=samples,null=null)\n    require(samples.shape==(n_boot,len(names),len(metrics)) and np.isfinite(samples).all(),\'Invalid allocation bootstrap cache\')\n    rows=[]\n    for ci,name in enumerate(names):\n        for mi,metric in enumerate(metrics):\n            lo,hi=np.quantile(samples[:,ci,mi],[.025,.975])\n            rows.append({\'study\':panel[\'study\'],\'component\':name,\'metric\':metric,\'estimate\':float(point[name][metric][0]),\n                \'ci_low\':float(lo),\'ci_high\':float(hi),\'ci_unit\':\'meaning; fixed languages/graph; percentile\'})\n    atomic_csv(output/\'component_variance_field_metrics.csv\',pd.DataFrame(rows))\n    lexical_edge=panel[\'Y\'].mean(0)@D;theta=np.linalg.pinv(B)@lexical_edge\n    coef=np.linalg.pinv(design(panel[\'pairs\'],panel[\'languages\'],\'both\'))@panel[\'Y\'].mean(0)\n    n=len(panel[\'languages\']);alpha=coef[1:1+n];beta=coef[1+n:1+2*n]\n    require(np.allclose(theta,alpha-beta,atol=1e-10),\'Potential/additive coefficient identity failed\')\n    atomic_csv(output/\'full450_language_coefficients.csv\',pd.DataFrame({\'study\':panel[\'study\'],\'language\':panel[\'languages\'],\n        \'alpha\':alpha,\'beta\':beta,\'theta\':theta}))\n    observed=float(point[\'lexical\'][\'gradient_fraction\'][0])\n    atomic_json(output/\'edge_permutation_null.json\',{\'observed_gradient_fraction\':observed,\'null_mean\':float(null.mean()),\n        \'null_q95\':float(np.quantile(null,.95)),\'p_plus_one\':float((1+(null>=observed).sum())/(n_boot+1)),\n        \'repetitions\':n_boot,\'null\':signature[\'null\'],\n        \'limit\':\'Descriptive edge-exchangeability null; not a model of sentence noise or new-language generalization.\'})\n    return pd.DataFrame(rows)\n'
ENVIRONMENT_LOCK='# This file was autogenerated by uv via the following command:\n#    uv pip compile work/e3_requirements.txt --python C:/Users/mahes/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe --python-version 3.12 --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match --output-file work/e3_environment_lock.txt\naccelerate==1.6.0\n    # via -r work/e3_requirements.txt\naiohappyeyeballs==2.7.1\n    # via aiohttp\naiohttp==3.14.5\n    # via fsspec\naiosignal==1.4.0\n    # via aiohttp\nattrs==26.1.0\n    # via aiohttp\ncertifi==2026.10.23\n    # via requests\ncharset-normalizer==3.5.2\n    # via requests\ndatasets==4.0.0\n    # via -r work/e3_requirements.txt\ndill==0.3.8\n    # via\n    #   datasets\n    #   multiprocess\nfilelock==3.32.7\n    # via\n    #   -r work/e3_requirements.txt\n    #   datasets\n    #   huggingface-hub\n    #   torch\n    #   transformers\n    #   triton\nfrozenlist==1.8.0\n    # via\n    #   aiohttp\n    #   aiosignal\nfsspec==2025.3.0\n    # via\n    #   datasets\n    #   huggingface-hub\n    #   torch\nhuggingface-hub==0.30.2\n    # via\n    #   -r work/e3_requirements.txt\n    #   accelerate\n    #   datasets\n    #   tokenizers\n    #   transformers\nidna==3.20\n    # via\n    #   requests\n    #   yarl\njinja2==3.1.6\n    # via torch\nmarkupsafe==3.0.4\n    # via jinja2\nmpmath==1.3.0\n    # via sympy\nmultidict==7.2.1\n    # via\n    #   aiohttp\n    #   yarl\nmultiprocess==0.70.16\n    # via datasets\nnetworkx==3.7\n    # via\n    #   -r work/e3_requirements.txt\n    #   torch\nnumpy==1.26.4\n    # via\n    #   -r work/e3_requirements.txt\n    #   accelerate\n    #   datasets\n    #   pandas\n    #   scipy\n    #   transformers\nnvidia-cublas-cu12==12.1.3.1\n    # via\n    #   nvidia-cudnn-cu12\n    #   nvidia-cusolver-cu12\n    #   torch\nnvidia-cuda-cupti-cu12==12.1.105\n    # via torch\nnvidia-cuda-nvrtc-cu12==12.1.105\n    # via torch\nnvidia-cuda-runtime-cu12==12.1.105\n    # via torch\nnvidia-cudnn-cu12==9.1.0.70\n    # via torch\nnvidia-cufft-cu12==11.0.2.54\n    # via torch\nnvidia-curand-cu12==10.3.2.106\n    # via torch\nnvidia-cusolver-cu12==11.4.5.107\n    # via torch\nnvidia-cusparse-cu12==12.1.0.106\n    # via\n    #   nvidia-cusolver-cu12\n    #   torch\nnvidia-nccl-cu12==2.21.5\n    # via torch\nnvidia-nvjitlink-cu12==12.1.105\n    # via\n    #   -r work/e3_requirements.txt\n    #   nvidia-cusolver-cu12\n    #   nvidia-cusparse-cu12\nnvidia-nvtx-cu12==12.1.105\n    # via torch\npackaging==26.3\n    # via\n    #   accelerate\n    #   datasets\n    #   huggingface-hub\n    #   transformers\npandas==2.2.3\n    # via\n    #   -r work/e3_requirements.txt\n    #   datasets\npropcache==0.5.4\n    # via\n    #   aiohttp\n    #   yarl\nprotobuf==4.25.3\n    # via -r work/e3_requirements.txt\npsutil==7.2.2\n    # via accelerate\npyarrow==20.0.0\n    # via\n    #   -r work/e3_requirements.txt\n    #   datasets\npython-dateutil==2.9.0.post0\n    # via\n    #   -r work/e3_requirements.txt\n    #   pandas\npytz==2026.6\n    # via pandas\npyyaml==6.0.3\n    # via\n    #   accelerate\n    #   datasets\n    #   huggingface-hub\n    #   transformers\nregex==2026.9.29\n    # via transformers\nrequests==2.34.2\n    # via\n    #   datasets\n    #   huggingface-hub\n    #   transformers\nsafetensors==0.5.3\n    # via\n    #   -r work/e3_requirements.txt\n    #   accelerate\n    #   transformers\nscipy==1.13.1\n    # via -r work/e3_requirements.txt\nsentencepiece==0.2.0\n    # via -r work/e3_requirements.txt\nsetuptools==75.8.0\n    # via\n    #   -r work/e3_requirements.txt\n    #   torch\nsix==1.17.0\n    # via python-dateutil\nsympy==1.13.1\n    # via torch\ntokenizers==0.21.1\n    # via\n    #   -r work/e3_requirements.txt\n    #   transformers\ntorch==2.5.1+cu121\n    # via\n    #   -r work/e3_requirements.txt\n    #   accelerate\ntqdm==4.70.1\n    # via\n    #   datasets\n    #   huggingface-hub\n    #   transformers\ntransformers==4.51.3\n    # via -r work/e3_requirements.txt\ntriton==3.1.0\n    # via torch\ntyping-extensions==4.16.0\n    # via\n    #   aiohttp\n    #   aiosignal\n    #   huggingface-hub\n    #   torch\ntzdata==2026.6\n    # via pandas\nurllib3==2.8.0\n    # via requests\nxxhash==4.0.2\n    # via datasets\nyarl==1.25.1\n    # via aiohttp\n'
WORKER_SHA256=sha256_bytes(WORKER_SOURCE.encode())
ANALYSIS_SHA256=sha256_bytes((E1_SOURCE+ANALYSIS_SOURCE).encode())
assert N_BOOT==PROTOCOL['bootstrap']['repetitions']
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
freeze_json(OUTPUT_ROOT/'protocol.json',PROTOCOL)
freeze_json(OUTPUT_ROOT/'meaning_split.json',MEANING_SPLIT)
freeze_json(OUTPUT_ROOT/'source_identity.json',{'worker_sha256':WORKER_SHA256,'analysis_sha256':ANALYSIS_SHA256})
original_split=validate_root(FP32_ROOT,EXPECTED,MEANING_SPLIT['original_split_sha256'])
PANELS={study:load_fp32_panel(FP32_ROOT,study,EXPECTED,original_split) for study in ['it2','nllb']}
assert PANELS['it2']['ids']==PANELS['nllb']['ids'] and PANELS['it2']['pairs']==PANELS['nllb']['pairs']
assert PANELS['it2']['corpus_ids']==PANELS['nllb']['corpus_ids']
manifest=pd.read_csv(FP32_ROOT/'sample_identity.csv',dtype={'corpus_sentence_id':str})
manifest=manifest[manifest.sentence_id.isin(PANELS['it2']['ids'])].sort_values(['pair','sentence_id'])
assert len(manifest)==18000
manifest_path=STAGING/'collection_identity.csv';atomic_csv(manifest_path,manifest)
for name,source in [('e1_core.py',E1_SOURCE),('e3_worker.py',WORKER_SOURCE),('e3_analysis.py',ANALYSIS_SOURCE),('environment_lock.txt',ENVIRONMENT_LOCK)]:
    (STAGING/name).write_text(source,encoding='utf-8')
    (OUTPUT_ROOT/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(STAGING))
from e3_analysis import panel_from_collector,supplementary
progress('Original baseline450 identities and both complete fp32 panels validated.')


# %% original code cell 3; notebook cell 6
# 3. Worker configuration and completion check.
COLLECTOR=OUTPUT_ROOT/'madlad_collector'
TEXT_CANDIDATES=[str(OUTPUT_ROOT/'private_text_cache/flores_source_texts.csv'),
 '/content/drive/MyDrive/indic_40_enhancements/E2_quality_comet_v1/private_text_cache/flores_source_texts.csv',
 '/content/drive/MyDrive/indic_40_downstream_strong/flores_source_texts.csv']
WORKER_CONFIG={'seed':20261105,'output':str(COLLECTOR),'model':MODEL_SPEC,'model_cache':str(MODEL_CACHE),
 'worker_sha256':WORKER_SHA256,'manifest':str(manifest_path),'manifest_sha256':sha256_file(manifest_path),
 'languages':sorted(LANG_CODES),'language_codes':LANG_CODES,'expected_tag_ids':TAG_IDS,
 'text_candidates':TEXT_CANDIDATES,'dataset':'openlanguagedata/flores_plus',
 'dataset_revision':'5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06','max_tokens':PROTOCOL['length_safety_ceiling']}
atomic_json(STAGING/'worker_config.json',WORKER_CONFIG)
def collector_complete():
    p=COLLECTOR/'complete.json'
    if not p.is_file():return False
    stamp=json.loads(p.read_text());lock=json.loads((COLLECTOR/'run_lock.json').read_text())
    require(stamp['worker_sha256']==WORKER_SHA256 and lock['model']==MODEL_SPEC,'Existing collector identity differs')
    require(lock['manifest_sha256']==WORKER_CONFIG['manifest_sha256'],'Existing collector meanings differ')
    require(stamp['run_lock_sha256']==sha256_file(COLLECTOR/'run_lock.json'),'Saved collector lock changed')
    require(stamp['all_rows_sha256']==sha256_file(COLLECTOR/'all_rows.csv'),'Completed collector rows changed')
    return True
def run_live(command,environment=None,log_name='worker.log'):
    process=subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,errors='replace')
    lines=queue.Queue()
    def reader():
        try:
            for line in process.stdout:lines.put(line)
        finally:lines.put(None)
    threading.Thread(target=reader,daemon=True).start();started=time.monotonic();last=started
    try:
        with (STAGING/log_name).open('a',encoding='utf-8') as log:
            while True:
                try:
                    line=lines.get(timeout=1)
                    if line is None:break
                    print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                if time.monotonic()-last>=30:
                    progress(f'Process alive for {(time.monotonic()-started)/60:.1f} min; waiting for next stage output.')
                    last=time.monotonic()
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        raise


# %% original code cell 4; notebook cell 7
# 4. Install/check the isolated L4 worker. Skip when the collection is already complete.
worker_environment=os.environ.copy();worker_environment.pop('PYTHONPATH',None)
worker_environment['PYTHONNOUSERSITE']='1';worker_environment['HF_HUB_DISABLE_XET']='1'
worker_environment['PYTHONPATH']=str(STAGING)
WORKER_PYTHON=WORKER_ENV/'bin/python'
if not collector_complete():
    run_live(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],log_name='setup.log')
    token=os.environ.get('HF_TOKEN')
    try:token=userdata.get('HF_TOKEN') or token
    except Exception:pass
    if token:worker_environment['HF_TOKEN']=token
    require(any(Path(p).is_file() for p in TEXT_CANDIDATES) or bool(token),'Enable HF_TOKEN for the original gated FLORES corpus.')
    run_live([sys.executable,'-m','pip','install','--quiet','uv==0.9.7'],log_name='setup.log')
    UV=[sys.executable,'-m','uv']
    if not WORKER_PYTHON.is_file():
        run_live(UV+['python','install','3.12'],log_name='setup.log')
        run_live(UV+['venv','--python','3.12',str(WORKER_ENV)],log_name='setup.log')
    run_live(UV+['pip','install','--python',str(WORKER_PYTHON),
        '--extra-index-url','https://download.pytorch.org/whl/cu121','--index-strategy','unsafe-best-match',
        '-r',str(STAGING/'environment_lock.txt')],log_name='setup.log')
else:progress('Completed MADLAD rows verified; GPU setup and collection will be skipped.')


# %% original code cell 5; notebook cell 8
# 5. Pre-tokenization, 11-target fp32 smoke test, then checkpointed full TF collection.
if not collector_complete():
    progress('Starting MADLAD worker. Text/model downloads and token checks precede the smoke timer.')
    run_live([str(WORKER_PYTHON),'-u',str(STAGING/'e3_worker.py'),'--config',str(STAGING/'worker_config.json')],worker_environment)
require(collector_complete(),'MADLAD collection did not finish or verify')
madlad_rows=pd.read_csv(COLLECTOR/'all_rows.csv',dtype={'corpus_sentence_id':str})
PANELS['madlad']=panel_from_collector(madlad_rows,PANELS['it2'],COLLECTOR)
print('All three systems now have the same 450 × 40 fp32 TF panel.')
display(pd.DataFrame([json.loads((COLLECTOR/'smoke_report.json').read_text())]))


# %% original code cell 6; notebook cell 10
# 6. Three-model E1-compatible generalization analysis. Saved results and RNG checkpoints resume.
summaries=[];contrasts=[]
for study,panel in PANELS.items():
    progress(f'=== E3 GENERALIZATION {study.upper()} ===')
    operators,audit=build_operators(panel['pairs'],panel['languages'])
    folder=OUTPUT_ROOT/'analysis'/study;folder.mkdir(parents=True,exist_ok=True)
    atomic_csv(folder/'graph_rank_checks.csv',audit)
    summary,contrast=run_study(panel,MEANING_SPLIT,operators,folder/'generalization',N_BOOT,20261102,128,ANALYSIS_SHA256)
    summaries.append(summary);contrasts.append(contrast)
all_metrics=pd.concat(summaries,ignore_index=True);all_contrasts=pd.concat(contrasts,ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E3_generalization_metrics.csv',all_metrics)
atomic_csv(OUTPUT_ROOT/'E3_additive_vs_target_contrasts.csv',all_contrasts)
display(all_metrics[(all_metrics.protocol=='unseen_meanings_and_dyad') & all_metrics.method.isin(['target','both'])])


# %% original code cell 7; notebook cell 11
# 7. Component means, row variance, field energy and null from the exact same baseline450.
component_tables=[];results=[]
for study,panel in PANELS.items():
    if study=='madlad':
        d=madlad_rows.copy();d['language_control']=d.target_tag
    else:
        run='holdout' if study=='it2' else 'confirm'
        d=pd.concat([pd.read_csv(p,dtype={'corpus_sentence_id':str}) for p in sorted((FP32_ROOT/run/'baseline_tf').glob('*.csv'))],ignore_index=True)
        d['pair']=d.source+'__'+d.tag
        d['language_control']=d.source_tag+d.target_tag if study=='it2' else d.source_code
        if study=='nllb':d['unknown']=0.0  # Unknown is included in the saved lexical component, not measured separately.
    d['lexical_with_unknown']=d.lexical+d.unknown
    d['conditional_lexical']=d.lexical/(d.lexical+d.punctuation)
    columns=['sentence_id','pair','lexical','eos','language_control','punctuation','unknown','conditional_lexical','lexical_with_unknown']
    components=d[columns].copy();components['study']=study
    require(np.allclose(components.pivot(index='sentence_id',columns='pair',values='lexical').loc[panel['ids'],panel['pairs']].to_numpy(),panel['Y']), 'Component/LM panels differ')
    component_tables.append(components)
    results.append(supplementary(panel,components,OUTPUT_ROOT/'analysis'/study/'allocation',N_BOOT,20261106))
allocation=pd.concat(results,ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E3_component_variance_field_metrics.csv',allocation)
display(allocation[(allocation.component=='lexical') & allocation.metric.isin(['mean_mass','row_direction_share','gradient_fraction','nonconservative_residual_L2'])])
means=pd.concat([pd.read_csv(OUTPUT_ROOT/'analysis'/s/'allocation/component_means.csv') for s in PANELS],ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E3_component_means.csv',means)
display(means[means.component.isin(['lexical','eos','language_control','punctuation','unknown'])])


# %% original code cell 8; notebook cell 12
# 8. Compact figures and a release archive. Weights and private corpus texts stay out.
import matplotlib.pyplot as plt
coefs=pd.concat([pd.read_csv(OUTPUT_ROOT/'analysis'/s/'allocation/full450_language_coefficients.csv') for s in PANELS],ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E3_language_coefficients.csv',coefs)
fig,axes=plt.subplots(1,2,figsize=(14,5))
for study in PANELS:
    z=coefs[coefs.study==study].set_index('language').loc[PANELS['it2']['languages']]
    axes[0].plot(z.index,z.beta,marker='o',label=study)
axes[0].tick_params(axis='x',rotation=75);axes[0].set_ylabel('Centered target coefficient β');axes[0].legend()
z=all_metrics[(all_metrics.protocol=='unseen_meanings_and_dyad') & (all_metrics.method=='both')]
for scale in ['individual_rows','direction_means']:
    q=z[z.scale==scale].set_index('study').loc[list(PANELS)]
    axes[1].plot(q.index,q.R2_test_centered,marker='o',label=scale)
axes[1].set_ylabel('Strict held-out R²');axes[1].legend();fig.tight_layout()
fig.savefig(OUTPUT_ROOT/'E3_main_comparison.png',dpi=200);fig.savefig(OUTPUT_ROOT/'E3_main_comparison.pdf');plt.show()
atomic_json(OUTPUT_ROOT/'limitations.json',{
 'unknown_conventions':'IT2/MADLAD unknown excluded and saved separately; NLLB unknown retained in original lexical component; recorded unknown zero is not a measured zero rate.',
 'cross_model':'Compare within-system structure. Absolute attention allocations depend on tokenizer, prefix positions, architecture and target support.',
 'scope':'Same previously examined corpus; 11 languages; 40 observed directions; no additional independent corpus or language sample.',
 'intervals':'Meaning-clustered, fixed graph/languages; generalization refits training; allocation and edge null descriptive.',
 'quality':'No generation/translation-quality result is established by this attention collection.'})
import zipfile
included=[]
for p in OUTPUT_ROOT.rglob('*'):
    if not p.is_file() or p.suffix in ['.tmp','.zip'] or p.name=='completed_all.json':continue
    rel=p.relative_to(OUTPUT_ROOT)
    if any(part in ['private_text_cache','parts','__pycache__'] for part in rel.parts):continue
    included.append(p)
atomic_json(OUTPUT_ROOT/'completed_all.json',{'experiment':'E3','models':list(PANELS),'rows_per_model':18000,
 'bootstrap_repetitions':N_BOOT,'worker_sha256':WORKER_SHA256,'analysis_sha256':ANALYSIS_SHA256,
 'outputs':{str(p.relative_to(OUTPUT_ROOT)):sha256_file(p) for p in included if p.name!='completed_all.json'}})
archive=OUTPUT_ROOT.with_name(OUTPUT_ROOT.name+'_results.zip');tmp=archive.with_suffix('.tmp.zip')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in included+[OUTPUT_ROOT/'completed_all.json']:
        z.write(p,str(p.relative_to(OUTPUT_ROOT)))
tmp.replace(archive)
print('E3 COMPLETE:',OUTPUT_ROOT)
print('Share E3_generalization_metrics.csv, E3_component_variance_field_metrics.csv and completed_all.json.')
print('Result archive:',archive)

