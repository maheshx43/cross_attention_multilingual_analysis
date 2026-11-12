# Original executed notebook code, in original cell order.
# Requires Colab/IPython and original inputs; see docs/VALIDATION_EXPERIMENTS.md.

# %% original code cell 0; notebook cell 1
# 1. CPU setup and paths. No pip installs or model/tokenizer downloads.
import os, platform, time, json, hashlib
from pathlib import Path
try:
    from IPython.display import display
except ImportError:
    display=print
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
    print('Local CPU runtime: set E1_FP32_ROOT / E1_OUTPUT_ROOT or edit the paths below.')
FP32_ROOT=Path(os.environ.get('E1_FP32_ROOT',
    '/content/drive/MyDrive/fp32_tag_intervention' if IN_COLAB else str(Path.cwd()/'data/fp32_tag_intervention')))
OUTPUT_ROOT=Path(os.environ.get('E1_OUTPUT_ROOT',
    '/content/drive/MyDrive/indic_40_enhancements/E1_generalization_cpu_v1' if IN_COLAB else str(Path.cwd()/'e1_generalization_outputs')))
N_BOOT=10_000
BOOT_SEED=20261102
BOOT_BATCH_SIZE=128
assert N_BOOT==10_000, 'This is the frozen final-report protocol. Use a separately labelled output/protocol for development changes.'
print('CPU-only input:',FP32_ROOT)
print('New output directory:',OUTPUT_ROOT)
print('Both models; 315/67/68 meanings; 10,000 whole-meaning bootstrap draws.')


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



# %% original code cell 2; notebook cell 4
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



# %% original code cell 3; notebook cell 6
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



# %% original code cell 4; notebook cell 8
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



# %% original code cell 5; notebook cell 11
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



# %% original code cell 6; notebook cell 13
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


# %% original code cell 7; notebook cell 14
# 2. Frozen protocol and exact input hashes, embedded for a self-contained notebook.
FROZEN_SPLIT=json.loads('{"seed": 20261101, "train": [0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 26, 27, 30, 33, 42, 45, 47, 51, 54, 56, 57, 58, 63, 64, 65, 68, 70, 72, 73, 75, 78, 79, 80, 82, 83, 85, 89, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 104, 108, 111, 112, 113, 121, 122, 124, 125, 126, 127, 128, 132, 133, 135, 136, 138, 142, 147, 150, 153, 154, 157, 158, 162, 163, 164, 166, 167, 168, 170, 172, 174, 175, 177, 178, 181, 186, 188, 195, 196, 197, 198, 199, 201, 203, 205, 206, 208, 211, 214, 218, 222, 225, 226, 228, 231, 235, 236, 237, 241, 243, 246, 248, 249, 251, 254, 255, 257, 259, 261, 263, 264, 266, 268, 271, 276, 278, 279, 283, 284, 285, 288, 290, 295, 299, 302, 303, 307, 309, 313, 314, 316, 318, 319, 321, 325, 327, 328, 330, 336, 337, 341, 342, 344, 346, 348, 352, 355, 357, 358, 359, 365, 366, 369, 370, 372, 376, 377, 386, 388, 389, 395, 398, 399, 400, 402, 403, 404, 409, 411, 412, 413, 415, 422, 427, 432, 433, 434, 437, 440, 441, 444, 448, 452, 456, 457, 459, 463, 467, 468, 470, 471, 472, 479, 484, 486, 490, 492, 495, 498, 502, 503, 508, 509, 510, 517, 520, 523, 525, 527, 528, 534, 535, 537, 542, 544, 546, 548, 551, 552, 555, 559, 561, 563, 564, 565, 568, 571, 575, 577, 579, 580, 582, 583, 584, 586, 588, 591, 592, 596, 605, 606, 608, 610, 612, 613, 615, 616, 618, 627, 628, 630, 632, 633, 634, 636, 637, 638, 639, 643, 645, 648, 651, 654, 656, 657, 658, 660, 661, 663, 667, 668, 669, 673, 677, 682, 683, 684, 685, 687, 689, 690, 692, 693, 696, 701, 702, 706, 707, 708, 710, 711, 712, 717, 718, 719, 722, 724, 728, 735, 739, 740, 743, 745], "validation": [5, 19, 28, 32, 52, 61, 69, 86, 120, 140, 151, 185, 219, 227, 252, 256, 269, 273, 298, 300, 323, 329, 331, 332, 345, 367, 374, 378, 380, 383, 393, 414, 428, 439, 442, 447, 475, 494, 512, 515, 530, 541, 549, 585, 603, 604, 622, 623, 631, 647, 650, 652, 655, 670, 674, 675, 676, 678, 697, 698, 705, 731, 733, 734, 737, 741, 748], "test": [8, 17, 24, 37, 40, 41, 49, 71, 77, 81, 87, 88, 93, 99, 106, 117, 130, 156, 165, 176, 190, 194, 200, 204, 207, 229, 232, 234, 245, 247, 253, 267, 274, 275, 281, 291, 305, 306, 315, 347, 351, 368, 375, 387, 391, 419, 426, 464, 474, 501, 511, 538, 595, 597, 621, 641, 649, 672, 681, 699, 703, 713, 723, 725, 726, 742, 746, 747], "original_split_sha256": "9ed17cbd0b4b2edd97e79bf3cd074af5929ee6ea98552ad0965f58afdaef43ef", "original_sample_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "scope": "450 original fp32 baseline meanings only"}')
FROZEN_PROTOCOL=json.loads('{"experiment": "E1", "protocol_version": "1.0", "models": ["it2", "nllb"], "input": "existing fp32 450-meaning teacher-forced baseline; no new inference; no old fp16 predictors", "status": "retrospective held-out validation of previously examined data; not a new untouched confirmatory corpus", "split": {"seed": 20261101, "train": [0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 26, 27, 30, 33, 42, 45, 47, 51, 54, 56, 57, 58, 63, 64, 65, 68, 70, 72, 73, 75, 78, 79, 80, 82, 83, 85, 89, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 104, 108, 111, 112, 113, 121, 122, 124, 125, 126, 127, 128, 132, 133, 135, 136, 138, 142, 147, 150, 153, 154, 157, 158, 162, 163, 164, 166, 167, 168, 170, 172, 174, 175, 177, 178, 181, 186, 188, 195, 196, 197, 198, 199, 201, 203, 205, 206, 208, 211, 214, 218, 222, 225, 226, 228, 231, 235, 236, 237, 241, 243, 246, 248, 249, 251, 254, 255, 257, 259, 261, 263, 264, 266, 268, 271, 276, 278, 279, 283, 284, 285, 288, 290, 295, 299, 302, 303, 307, 309, 313, 314, 316, 318, 319, 321, 325, 327, 328, 330, 336, 337, 341, 342, 344, 346, 348, 352, 355, 357, 358, 359, 365, 366, 369, 370, 372, 376, 377, 386, 388, 389, 395, 398, 399, 400, 402, 403, 404, 409, 411, 412, 413, 415, 422, 427, 432, 433, 434, 437, 440, 441, 444, 448, 452, 456, 457, 459, 463, 467, 468, 470, 471, 472, 479, 484, 486, 490, 492, 495, 498, 502, 503, 508, 509, 510, 517, 520, 523, 525, 527, 528, 534, 535, 537, 542, 544, 546, 548, 551, 552, 555, 559, 561, 563, 564, 565, 568, 571, 575, 577, 579, 580, 582, 583, 584, 586, 588, 591, 592, 596, 605, 606, 608, 610, 612, 613, 615, 616, 618, 627, 628, 630, 632, 633, 634, 636, 637, 638, 639, 643, 645, 648, 651, 654, 656, 657, 658, 660, 661, 663, 667, 668, 669, 673, 677, 682, 683, 684, 685, 687, 689, 690, 692, 693, 696, 701, 702, 706, 707, 708, 710, 711, 712, 717, 718, 719, 722, 724, 728, 735, 739, 740, 743, 745], "validation": [5, 19, 28, 32, 52, 61, 69, 86, 120, 140, 151, 185, 219, 227, 252, 256, 269, 273, 298, 300, 323, 329, 331, 332, 345, 367, 374, 378, 380, 383, 393, 414, 428, 439, 442, 447, 475, 494, 512, 515, 530, 541, 549, 585, 603, 604, 622, 623, 631, 647, 650, 652, 655, 670, 674, 675, 676, 678, 697, 698, 705, 731, 733, 734, 737, 741, 748], "test": [8, 17, 24, 37, 40, 41, 49, 71, 77, 81, 87, 88, 93, 99, 106, 117, 130, 156, 165, 176, 190, 194, 200, 204, 207, 229, 232, 234, 245, 247, 253, 267, 274, 275, 281, 291, 305, 306, 315, 347, 351, 368, 375, 387, 391, 419, 426, 464, 474, 501, 511, 538, 595, 597, 621, 641, 649, 672, 681, 699, 703, 713, 723, 725, 726, 742, 746, 747], "original_split_sha256": "9ed17cbd0b4b2edd97e79bf3cd074af5929ee6ea98552ad0965f58afdaef43ef", "original_sample_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "scope": "450 original fp32 baseline meanings only"}, "fit_meanings": "train315 only; validation is reported, not used for tuning or refitting", "protocols": {"unseen_meanings": "fit all 40 directions on train315; evaluate test68 in the same directions", "unseen_dyad_shared_meanings": "fit train315 on 38 directions; evaluate both omitted directions on the same train315 meanings; diagnostic only", "unseen_meanings_and_dyad": "fit train315 on 38 directions; evaluate both omitted directions on test68; primary strict generalization"}, "models_compared": ["mean", "source", "target", "both", "direction_mean"], "direction_mean_unseen_dyad_policy": "not estimable; report unavailable without imputing a fallback", "primary_scales": ["individual_rows", "direction_means"], "metrics": ["R2_test_centered", "R2_training_mean_reference", "MAE", "RMSE"], "per_dyad_reporting": "all 20 dyads, both directions excluded; row and direction-mean MAE/RMSE; do not interpret two-point fold R2", "graph_checks": "all retained graphs connected; source/target design full identifiable rank after each deletion", "bootstrap": {"repetitions": 10000, "seed": 20261102, "unit": "whole meanings jointly across all directions", "training_uncertainty": "refit every draw via fixed linear fit operators", "independent_meaning_protocols": "independently resample train315 and test68", "shared_meaning_protocol": "same resampled meaning weights in training and held-dyad evaluation", "between_model_pairing": "same resampling seed and aligned meaning IDs for both systems", "interval": "percentile 95%; graph, languages and selected corpus fixed"}, "primary_contrast": "both versus target on identical strict-held-out rows; report R2 difference and reductions in MAE/RMSE", "exclusions": "require complete eligible 450x40 fp32 TF panels; stop on missing, capped, duplicate, mixed-tag/text or mismatched rows; no zero imputation", "no_significance_claim_for_language_generalization": "meaning intervals do not quantify uncertainty over new languages/models/corpora", "no_github_changes": true}')
EXPECTED=json.loads('{"files": {"it2": {"asm_Beng__ben_Beng__ben_Beng.csv": "4634845e28c39d95e7c8130971d5cfd11747e3e158e2da156ab3571e94335e67", "asm_Beng__kan_Knda__kan_Knda.csv": "81c88e00168ca2259e6787e77135ebd436072a3645d06d16af4dc031965c6b46", "ben_Beng__asm_Beng__asm_Beng.csv": "6ebb30daf35a4991f9b8b2bede5290ec1dcd33c3b2abe319b7847e0deb7576de", "ben_Beng__kan_Knda__kan_Knda.csv": "30b6b14f9dc8d4bdd7d6c54585e339428621be1f4a2fe7ab165252c7bbe74ea7", "ben_Beng__mar_Deva__mar_Deva.csv": "3bbbcdf86d1226466072057efdee555b4b0b806f902cb613cd4363622a23b855", "ben_Beng__tam_Taml__tam_Taml.csv": "c40518929585fde76be7ba987261ca167f77ea88271117dcd5c2f40605506566", "guj_Gujr__mal_Mlym__mal_Mlym.csv": "ce14cd4c07ec11e8137aabfff61ea2738bbbee03039fe9273ac56ce409383c78", "guj_Gujr__mar_Deva__mar_Deva.csv": "f853aa655b03c6cf0f2446c9c0d3980e73d2909a87f36a881bde5d7efdb86756", "guj_Gujr__pan_Guru__pan_Guru.csv": "72a2cb9e92658689eafc36ce542117b3ff4fe5a82f68b3e369b480cfab0411c1", "hin_Deva__kan_Knda__kan_Knda.csv": "0563a0404498980d2c98c9a176195989bfbbf271631ca942be1456fbf3940500", "hin_Deva__mal_Mlym__mal_Mlym.csv": "35e5193ac09a23882035688bd2fce64dc546470d128e3a201f4e754df0fd0bda", "hin_Deva__ory_Orya__ory_Orya.csv": "1a7eaaefeb09bfbcc753d96f664b38d3ac7e1dcc5aa29f91db01880345ecdb93", "hin_Deva__tam_Taml__tam_Taml.csv": "bc1a5e084215749a262e753743f9a740f1d334ddd6c7ff7410c77b2628235301", "kan_Knda__asm_Beng__asm_Beng.csv": "a9ca741ff9a9d3de7b74607f8ee4f87b6e8086d0b245ab49565f3c6489191212", "kan_Knda__ben_Beng__ben_Beng.csv": "3621e04c235045fb9c910afe57a20676593726e0b1480aaad56d5d389a771297", "kan_Knda__hin_Deva__hin_Deva.csv": "6ca052a4d5837fba7ec001e2ecd32317257fb0913610a7d1bb11a1a3ff4c96f2", "kan_Knda__mal_Mlym__mal_Mlym.csv": "b02df5f356eb72d2c8a38d551c0eec3af90d1035ad7db83f981544c560e76650", "kan_Knda__mar_Deva__mar_Deva.csv": "84776eed18969cfee5a7b1823bf043999e0983b90d0a046f6ae9bcf018d9e06a", "kan_Knda__tam_Taml__tam_Taml.csv": "bfc457c1dc99189a070cae1bf517f2e2bbc7dc9b9517bf4333cdc7bb612e5b6c", "mal_Mlym__guj_Gujr__guj_Gujr.csv": "bb9e390740e851548d35f92b580c580904b3bce71a042a3d9ee65a8a2d4f8fcb", "mal_Mlym__hin_Deva__hin_Deva.csv": "7fa15dc5e9f999d6c6a7cd936f1e0c3eea3e890310b06424f69c21249b255645", "mal_Mlym__kan_Knda__kan_Knda.csv": "c2a67cdb13447efea979449c2b232b2fd3c1c87e3d68c0995e36d81034eb6569", "mal_Mlym__tam_Taml__tam_Taml.csv": "7a6de872b1df6d2145bc0c8ac15e16d9380e9e4c51fb4c7e49ff43891210669d", "mal_Mlym__tel_Telu__tel_Telu.csv": "ec7db98f64b79552351d435c8364bc1e0b4974f0a5d772532c98e1a5634c20e7", "mar_Deva__ben_Beng__ben_Beng.csv": "1342491709a2464c2a4d50757da6ca52b6383b95b95989456870f31e881945d8", "mar_Deva__guj_Gujr__guj_Gujr.csv": "64560ef6135751c1a5b30f98dc16b1b60f5414ff87bc248516fea5a3100c119a", "mar_Deva__kan_Knda__kan_Knda.csv": "27752e44123549dd39b4b3d6b39e8acb37d1d6f733bbbb54cf42ecc3d08387aa", "ory_Orya__hin_Deva__hin_Deva.csv": "2bcdf4c178b1fdf44bcdd5a3f1800e2355722dd21cb5c4621afe34a0e3cec981", "ory_Orya__tel_Telu__tel_Telu.csv": "668ba745204726558eac09e21b8bdc8a7ee14cbefa34c7a7ab3547f26727f651", "pan_Guru__guj_Gujr__guj_Gujr.csv": "f75cf9bcfd8954ebc0b5049577a15bc233f45a0abcb49cbaaaedfb02b517ed4a", "pan_Guru__tam_Taml__tam_Taml.csv": "84a06227916ef7155c3e8da7f832cd7a95961cf81524c143380b09c1063fd4d2", "tam_Taml__ben_Beng__ben_Beng.csv": "acd49c2d8d155f060391808ee76a17d2a4d6ae8c64ec2329d6361f763d0207e6", "tam_Taml__hin_Deva__hin_Deva.csv": "8db4da21c070a11e0e69441e3e070400101f72b102ed27e55fc317818e6cb18e", "tam_Taml__kan_Knda__kan_Knda.csv": "c2d1c55f4fc586deecf37da432c1433f92fd128cb275da18ebb84637961753b2", "tam_Taml__mal_Mlym__mal_Mlym.csv": "4cac02d01a1292ad5a9f52dc003632fc59ecde6cfd68df3ce1210dfacdc10400", "tam_Taml__pan_Guru__pan_Guru.csv": "50b8d0cafdd44ce91d21e0f17ecddd1269ccfda80c833a3aae6387fb2cfe45da", "tam_Taml__tel_Telu__tel_Telu.csv": "b76d1fade10dc79e5b0bcfe44618780a9ad4e72850f5582bf0d73353c2ce75bc", "tel_Telu__mal_Mlym__mal_Mlym.csv": "df3b7400b553b79c2a55d71b4dec630eb58e77df66b16bf1467393e62ccca932", "tel_Telu__ory_Orya__ory_Orya.csv": "b658578c14b9382c979b903143f373932ca1efc40599350910bc9cd5588ca73f", "tel_Telu__tam_Taml__tam_Taml.csv": "e53e5ccbba0ec047d6bb717a2a65982820ae8e8e9c3172270b27da2816687188"}, "nllb": {"asm_Beng__ben_Beng__ben_Beng.csv": "2064bc6293a2865309e16516071df5c59022b912e16c96fdab5fd4b283cff36e", "asm_Beng__kan_Knda__kan_Knda.csv": "b577608427097fb2a9a00ae64ecddbdd38266e690dc0813509500fa28f7a92a6", "ben_Beng__asm_Beng__asm_Beng.csv": "c151e1db1ddfd6fcbab4c87481f416fcb5cea4b18989d8717e1c35fa2bd453cd", "ben_Beng__kan_Knda__kan_Knda.csv": "b8cd209b383a759a6fe1fd8cc91ad5083dc0bbdb37e5c20d5997bfecedf9d904", "ben_Beng__mar_Deva__mar_Deva.csv": "bea2e02895ef5f8f10531916970e6121755ea6504c9234f3b9989163ba7fc6e0", "ben_Beng__tam_Taml__tam_Taml.csv": "34dd6305e12e258a82fbc0c2d6f2e623be73ac0df0736be104219c2d06d950f8", "guj_Gujr__mal_Mlym__mal_Mlym.csv": "0acbda7063180c28f70c5be357c1acbc0456771842c1206e15c802d55046610b", "guj_Gujr__mar_Deva__mar_Deva.csv": "743b82bbe6ae411cb78b7c4dcef64800c2373c73acaf58c5a390a7e02e076ff4", "guj_Gujr__pan_Guru__pan_Guru.csv": "f0e7f433bb0ee10ac03a95c6f4cf2e4b08b1b0e0ef1c1ba558a6499cf2d1de8d", "hin_Deva__kan_Knda__kan_Knda.csv": "1097f756f9fe0630594cec12c4ab5d501f25c53c9034fb0f743228f1149af268", "hin_Deva__mal_Mlym__mal_Mlym.csv": "18e4d46bbf17f256f7eda806ca18433c722084cebc72cb7dbeb5466c67f49050", "hin_Deva__ory_Orya__ory_Orya.csv": "f21b16f1e446b7536517a02b212e72796f0dcea09e5d339e521b50bb6b3e5767", "hin_Deva__tam_Taml__tam_Taml.csv": "5da204353dad25cac0925c7737bf4468dea22f09eafae485ac8c090272c48310", "kan_Knda__asm_Beng__asm_Beng.csv": "f9a543bc88d15281b36ea46195928dfa4303f07dd5c2b9b75cc2910596dcd18b", "kan_Knda__ben_Beng__ben_Beng.csv": "47702404293386f00e3bc6fb43df22bd64716e6cf250cc9ff2c44db0b98bfd28", "kan_Knda__hin_Deva__hin_Deva.csv": "6aabeed62e2aed01e293118e63834b53a2ad8c7432c8002ed2ae6c0f97703b3b", "kan_Knda__mal_Mlym__mal_Mlym.csv": "886c84e25998f1994666c2da333635dd767bc89e7c9ea4b336971fff951c4d17", "kan_Knda__mar_Deva__mar_Deva.csv": "813f21e381342c9a1e5a9bcc91e548e3760cb29708c2d263170f9c588215a575", "kan_Knda__tam_Taml__tam_Taml.csv": "46a9d60b3ed4ac6c3574e7572b5c6fe367edb8bbd9e0bfeeaf530f576ebfedbd", "mal_Mlym__guj_Gujr__guj_Gujr.csv": "609980fbfe47a4b3ba7426defe6a9544867d17a809acde707b2364a0bfbecf5f", "mal_Mlym__hin_Deva__hin_Deva.csv": "afa69e281e7b50335f4e4f216f76f57cf9db42791f8d1382e308d45aeff39570", "mal_Mlym__kan_Knda__kan_Knda.csv": "d7e378175aced0fae1d577750f52b13176bc2568a2892982b30972894e7e9f7b", "mal_Mlym__tam_Taml__tam_Taml.csv": "6c62596177c3a4bcdacd8c9f6bad05ebafd108b9843168c326528e0510a19b3a", "mal_Mlym__tel_Telu__tel_Telu.csv": "bfff8f20f5119253d905c1f9fc02c157d2f75b48584ded9d2bbf209191e163c7", "mar_Deva__ben_Beng__ben_Beng.csv": "ca28071d7a88b46fc228a0d6a69d1645f2e9ca04388685253922381cb0865dee", "mar_Deva__guj_Gujr__guj_Gujr.csv": "727d3e552df5a250577f13aeae48e696b7809f4b0de422f28557a8803f73c56b", "mar_Deva__kan_Knda__kan_Knda.csv": "3aa565d5c06332a91f3dab40ca1a36335c13ed812619a901cacf70f25e95f0f4", "ory_Orya__hin_Deva__hin_Deva.csv": "dcafc3a172acd79191107371f21019cbffed71b03e05960788ec7532697a8df6", "ory_Orya__tel_Telu__tel_Telu.csv": "3ccd6b7146e9dcf7dac782a726c5b082e56cc9c66cec59842081e202b81ef965", "pan_Guru__guj_Gujr__guj_Gujr.csv": "90236621d8121c30c315d2e621eb1e323c9d2242bf88449b147a61700dc16702", "pan_Guru__tam_Taml__tam_Taml.csv": "89e84bf644951fc9d2b9cd7989db5522cb17559a49a113e9fc8dc993e77546d9", "tam_Taml__ben_Beng__ben_Beng.csv": "ecfa082cbadaaf05fef587a246f213461e17c641070da62b6ecec43340f51548", "tam_Taml__hin_Deva__hin_Deva.csv": "37458e8f157e3b47f2cd71f7e87b100e18d12069489834e48ed2237e7ac8091c", "tam_Taml__kan_Knda__kan_Knda.csv": "292ba6bfdd962065032c97497861e5cc02318a5027cab7162a9cc91e879b7659", "tam_Taml__mal_Mlym__mal_Mlym.csv": "2bc67edf61d73b41edeb4d07adf90fad48b5bd2ffa335751cf8ca2663f3666d0", "tam_Taml__pan_Guru__pan_Guru.csv": "f6ca99718db5ccdad447e9d68dc65869012dbb7693026e57b50b14d8f373ca6e", "tam_Taml__tel_Telu__tel_Telu.csv": "3f3922a2af7a1b079f78588436b991aacc8da08cb814c3bd92afef386abfc5bd", "tel_Telu__mal_Mlym__mal_Mlym.csv": "a8bda76d64a618a89ebfb3dda4c60857fbb6b8771e023e4a5a2d0b03ca0c4e2f", "tel_Telu__ory_Orya__ory_Orya.csv": "30ae2f50cb771d050cf75524fa00dbaca65f8c968985e15b21caf2ba925b67f9", "tel_Telu__tam_Taml__tam_Taml.csv": "6c0f1cd62eff90c51d1c8ef632f3f86536fa50e2c27b1e88c1bedb40f51e79a5"}}, "run_locks": {"it2": {"model": "ai4bharat/indictrans2-indic-indic-dist-320M", "model_revision": "ffb7582b6d43791f1fb26b2153fc065f2e9ea575", "dataset": "openlanguagedata/flores_plus", "dataset_revision": "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06", "split_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "split_seed": 240902, "bootstrap_seed": 240903, "n_boot": 10000, "targets": ["pan_Guru", "mal_Mlym", "tam_Taml", "guj_Gujr", "hin_Deva", "ben_Beng"], "sources": ["kan_Knda", "tel_Telu", "ory_Orya", "mar_Deva"], "control": "urd_Arab", "generation_meanings": 150, "generation_ids_sha256": "a54f1be4a1069f3e0d7e08cbf7151bc05712a84306f63e3f1bf874776b084c32", "torch_version": "2.11.0+cu128", "transformers_version": "4.51.3"}, "nllb": {"model": "facebook/nllb-200-distilled-600M", "model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d", "dataset": "openlanguagedata/flores_plus", "dataset_revision": "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06", "split_hash": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c", "split_seed": 240902, "bootstrap_seed": 240903, "n_boot": 10000, "targets": ["pan_Guru", "mal_Mlym", "tam_Taml", "guj_Gujr", "hin_Deva", "ben_Beng"], "sources": ["kan_Knda", "tel_Telu", "ory_Orya", "mar_Deva"], "control": "urd_Arab", "generation_meanings": 150, "generation_ids_sha256": "a54f1be4a1069f3e0d7e08cbf7151bc05712a84306f63e3f1bf874776b084c32", "torch_version": "2.11.0+cu128", "transformers_version": "4.51.3"}}, "sample_sha256": "64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c"}')
DRIVE_AUDIT=json.loads('{"checked_on": "2026-11-01", "scope": "User-shared fp32 baseline/analysis, LOLO and row-variance folders relevant to E1", "finding": "Existing LOLO/LOEO and same-meaning row-variance outputs found; no E1 independent-meaning plus held-dyad output identified in checked folders. This is not a claim about unshared Drive locations.", "folders": [{"key": "holdout/baseline_tf", "files": 80, "csv_files": 40, "npz_files": 40}, {"key": "confirm/baseline_tf", "files": 80, "csv_files": 40, "npz_files": 40}, {"key": "holdout/analysis", "files": 7, "csv_files": 4, "npz_files": 0}, {"key": "confirm/analysis", "files": 7, "csv_files": 4, "npz_files": 0}, {"key": "variance", "files": 13, "csv_files": 10, "npz_files": 0}, {"key": "lolo/holdout", "files": 10, "csv_files": 6, "npz_files": 0}, {"key": "lolo/nllb", "files": 10, "csv_files": 6, "npz_files": 0}, {"key": "lolo/pilot", "files": 10, "csv_files": 6, "npz_files": 0}]}')
CORE_SOURCE_SHA256='43bd0136cd01cbc3fba8d61e756c76fe5ae0ed437bbc816f2770a981d7a66122'
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
freeze_json(OUTPUT_ROOT/'meaning_split.json',FROZEN_SPLIT)
freeze_json(OUTPUT_ROOT/'protocol.json',FROZEN_PROTOCOL)
freeze_json(OUTPUT_ROOT/'input_expectations.json',EXPECTED)
freeze_json(OUTPUT_ROOT/'drive_audit_snapshot.json',DRIVE_AUDIT)
RUN_SETTINGS={'analysis':'E1 saved fp32 generalization','version':'1.0',
    'code_sha256':CORE_SOURCE_SHA256,'n_boot':N_BOOT,'bootstrap_seed':BOOT_SEED,
    'bootstrap_batch_size':BOOT_BATCH_SIZE,'split_seed':FROZEN_SPLIT['seed'],
    'input_root':str(FP32_ROOT),'only_existing_fp32_TF_inputs':True}
freeze_json(OUTPUT_ROOT/'run_settings.json',RUN_SETTINGS)
atomic_json(OUTPUT_ROOT/'runtime_environment.json',{'python':platform.python_version(),
    'numpy':np.__version__,'pandas':pd.__version__,'cpu_only':True})
try:
    from threadpoolctl import threadpool_limits
    _cpu_thread_limit=threadpool_limits(limits=1)
except ImportError:
    pass
print('Private protocol and meaning split saved before any new evaluation scores.')
print('Train/validation/test:',*[len(FROZEN_SPLIT[k]) for k in ['train','validation','test']])


# %% original code cell 8; notebook cell 16
# 3. Read and validate inputs. No experimental score is computed in this cell.
original_split=validate_root(FP32_ROOT,EXPECTED,FROZEN_SPLIT['original_split_sha256'])
assert set(FROZEN_SPLIT['train']+FROZEN_SPLIT['validation']+FROZEN_SPLIT['test'])==set(original_split['baseline450'])
assert not set(FROZEN_SPLIT['train']+FROZEN_SPLIT['validation']+FROZEN_SPLIT['test']) & set(original_split['intervention300'])
PANELS={}
for study in ['it2','nllb']:
    PANELS[study]=load_fp32_panel(FP32_ROOT,study,EXPECTED,original_split)
    panel=PANELS[study]
    atomic_npz(OUTPUT_ROOT/study/'verified_input_panel.npz',LM=panel['Y'],
        sentence_id=np.array(panel['ids']),corpus_sentence_id=np.array(panel['corpus_ids']),
        directions=np.array(panel['pairs']))
    atomic_json(OUTPUT_ROOT/study/'input_provenance.json',{
        'study':study,'rows':18000,'meanings':450,'directions':40,'precision_of_collection':'fp32',
        'numeric_fitting_dtype':'float64','csv_sha256':panel['csv_sha256'],
        'run_lock_sha256':panel['run_lock_sha256'],'sample_manifest_sha256':EXPECTED['sample_sha256'],
        'original_split_sha256':FROZEN_SPLIT['original_split_sha256']})
assert PANELS['it2']['ids']==PANELS['nllb']['ids']
assert PANELS['it2']['pairs']==PANELS['nllb']['pairs']
assert PANELS['it2']['corpus_ids']==PANELS['nllb']['corpus_ids']
OPERATORS,GRAPH_AUDIT=build_operators(PANELS['it2']['pairs'],PANELS['it2']['languages'])
atomic_csv(OUTPUT_ROOT/'fold_design_audit.csv',GRAPH_AUDIT)
for panel in PANELS.values():split_indices(panel,FROZEN_SPLIT)
print('Verified: both complete 18,000-row fp32 panels, identical meaning identities and all 20 identifiable held-dyad folds.')
display(GRAPH_AUDIT.groupby('method')[['rank','expected_rank']].min())


# %% original code cell 9; notebook cell 18
# 4. Validation point estimates; no test labels are used here.
validation_tables=[]
for study,panel in PANELS.items():
    table,dyads,_=point_evaluation(panel,FROZEN_SPLIT,OPERATORS,'validation',False)
    atomic_csv(OUTPUT_ROOT/study/'validation_point_metrics.csv',table)
    atomic_csv(OUTPUT_ROOT/study/'validation_per_dyad.csv',dyads)
    validation_tables.append(table)
validation=pd.concat(validation_tables,ignore_index=True)
display(validation[validation.available & validation.method.isin(['target','both'])][
    ['study','protocol','method','scale','R2_test_centered','MAE','RMSE']])
print('Proceed with the frozen specifications; do not tune based on this table.')


# %% original code cell 10; notebook cell 20
# 5. Run both systems. Restarting this cell resumes verified checkpoints.
all_summaries=[];all_contrasts=[];draw_chains=[]
for study,panel in PANELS.items():
    progress(f'=== E1 {study.upper()} ===')
    summary,contrasts=run_study(panel,FROZEN_SPLIT,OPERATORS,OUTPUT_ROOT/study,
        N_BOOT,BOOT_SEED,BOOT_BATCH_SIZE,CORE_SOURCE_SHA256)
    all_summaries.append(summary);all_contrasts.append(contrasts)
    stamp=json.loads((OUTPUT_ROOT/study/'completed.json').read_text())
    assert stamp['bootstrap_completed']==N_BOOT
    draw_chains.append(stamp['bootstrap_draw_chain'])
assert len(set(draw_chains))==1,'Between-model bootstrap draws were not aligned.'
SUMMARY=pd.concat(all_summaries,ignore_index=True)
CONTRASTS=pd.concat(all_contrasts,ignore_index=True)
atomic_csv(OUTPUT_ROOT/'E1_all_test_metrics.csv',SUMMARY)
atomic_csv(OUTPUT_ROOT/'E1_additive_vs_target_contrasts.csv',CONTRASTS)
display(SUMMARY[SUMMARY.available & SUMMARY.method.isin(['target','both'])][
    ['study','protocol','method','scale','R2_test_centered','R2_test_centered_ci_low',
     'R2_test_centered_ci_high','MAE','RMSE']])
print('Primary comparison: additive versus target-only, strict unseen_meanings_and_dyad protocol.')
display(CONTRASTS[CONTRASTS.protocol.eq('unseen_meanings_and_dyad')][
    ['study','scale','metric','estimate','ci_low','ci_high','positive_means']])


# %% original code cell 11; notebook cell 21
# 6. Plot the saved estimates and intervals. Tables remain authoritative.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib unavailable: numerical results are complete; plotting skipped.')
else:
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    protocols=['unseen_meanings','unseen_meanings_and_dyad']
    for ri,study in enumerate(['it2','nllb']):
        for ci,scale in enumerate(SCALES):
            ax=axes[ri,ci]
            for method,color,offset in [('target','#d4962b',-.08),('both','#336da8',.08)]:
                rows=SUMMARY[SUMMARY.study.eq(study)&SUMMARY.scale.eq(scale)&SUMMARY.method.eq(method)].set_index('protocol')
                for j,p in enumerate(protocols):
                    row=rows.loc[p];x=j+offset
                    ax.vlines(x,row.R2_test_centered_ci_low,row.R2_test_centered_ci_high,color=color,lw=2)
                    ax.scatter(x,row.R2_test_centered,color=color,label=method if j==0 else None,zorder=3)
            ax.axhline(0,color='gray',lw=.7)
            ax.set_xticks([0,1],['New meanings','New meanings + dyad'])
            ax.set_title(f'{study.upper()} | {scale.replace("_"," ")}')
            ax.set_ylabel('Held-out R², evaluation-mean reference')
            ax.legend(frameon=False)
    fig.savefig(OUTPUT_ROOT/'E1_generalization.png',dpi=220)
    fig.savefig(OUTPUT_ROOT/'E1_generalization.pdf')
    display(fig);plt.close(fig)


# %% original code cell 12; notebook cell 23
# 7. Final completion record and a compact results archive; no GitHub changes.
atomic_json(OUTPUT_ROOT/'completed_all.json',{
    'experiment':'E1','models':['it2','nllb'],'train_meanings':315,'validation_meanings':67,'test_meanings':68,
    'bootstrap_repetitions':N_BOOT,'aligned_bootstrap_draw_chain':draw_chains[0],
    'status':'retrospective held-out validation; graph/languages/corpus fixed',
    'study_completion_sha256':{s:sha256_file(OUTPUT_ROOT/s/'completed.json') for s in PANELS},
    'combined_summary_sha256':sha256_file(OUTPUT_ROOT/'E1_all_test_metrics.csv'),
    'code_sha256':CORE_SOURCE_SHA256})
import zipfile
archive=OUTPUT_ROOT.parent/(OUTPUT_ROOT.name+'_results.zip')
temporary=archive.with_suffix('.tmp.zip')
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(OUTPUT_ROOT.rglob('*')):
        if path.is_file() and not path.name.endswith('.tmp'):
            zipped.write(path,path.relative_to(OUTPUT_ROOT).as_posix())
temporary.replace(archive)
print('E1 COMPLETE. Results:',OUTPUT_ROOT)
print('Result archive:',archive)
print('Share E1_all_test_metrics.csv, E1_additive_vs_target_contrasts.csv and completed_all.json for review.')

