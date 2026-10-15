# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.
from IPython import get_ipython

# %% original code cell 0; notebook cell 1
get_ipython().run_line_magic('pip', '-q install "numpy>=1.26,<2.5" "pandas>=2.1,<4" "scipy>=1.11,<2"')

from google.colab import drive
drive.mount('/content/drive')

import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from IPython.display import display


# %% original code cell 1; notebook cell 3
from pathlib import Path

DATA_ROOT = Path('/content/drive/MyDrive/indic_40_backups/nllb_40_confirmatory_holdout_750_final_v6')
OUTPUT_ROOT = Path('/content/drive/MyDrive/indic_40_conservative_field_final_stats/nllb_confirmatory')

CONFIG = dict(
    data_root=str(DATA_ROOT),
    output_root=str(OUTPUT_ROOT),
    dataset_label='nllb_confirmatory',
    expected_n_sentences=750,
    expected_model_substring='nllb',
    required_root_substring='confirmatory',
    forbidden_root_substrings=[],
    nllb_missingness=True,
    folds=5,
    seed=20260922,
    permutations=9999,
    min_paired=20,
)


# %% original code cell 2; notebook cell 5
MODES = ['teacher_forced', 'free_generation']

def safe_mean(x, axis=None):
    x = np.asarray(x, dtype=float)
    count = np.isfinite(x).sum(axis=axis)
    total = np.where(np.isfinite(x), x, 0.0).sum(axis=axis)
    return np.divide(
        total, count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0
    )

def _truthy(series):
    return series.astype(str).str.strip().str.lower().isin(
        ['1', '1.0', 'true', 'yes']
    )

def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def reverse_pair(pair):
    a, b = str(pair).split('__')
    return f'{b}__{a}'

def incidence_matrix(dyads):
    langs = sorted({lang for p, _ in dyads for lang in p.split('__')})
    inc = np.zeros((len(dyads), len(langs)), dtype=float)
    for j, (p, q) in enumerate(dyads):
        a, b = p.split('__')
        inc[j, langs.index(a)] = 1.0
        inc[j, langs.index(b)] = -1.0
    return langs, inc

def r2_against_zero(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return np.nan
    base = float(np.sum(y[ok] ** 2))
    return (
        float(1.0 - np.sum((y[ok] - pred[ok]) ** 2) / base)
        if base > 0 else np.nan
    )


# %% original code cell 3; notebook cell 7
def load_collection_context(cfg):
    root = Path(cfg['data_root']).expanduser()
    out = Path(cfg['output_root']).expanduser()

    assert root.is_dir(), f'Experiment folder unavailable: {root}'
    root_name = root.name.lower()
    assert cfg['required_root_substring'].lower() in root_name
    assert not any(x.lower() in root_name for x in cfg['forbidden_root_substrings'])

    required_paths = [
        root / 'teacher_forced_all_stats.csv',
        root / 'free_generation_all_stats.csv',
        root / 'manifests' / 'run_config.json',
        root / 'manifests' / 'sample_identity.csv',
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    assert not missing, f'Missing required GPU collection artifacts: {missing}'

    run_config = json.loads(
        (root / 'manifests' / 'run_config.json').read_text(encoding='utf-8')
    )
    assert int(run_config['n_sentences']) == int(cfg['expected_n_sentences'])
    assert cfg['expected_model_substring'].lower() in str(run_config['model']).lower()

    manifest = pd.read_csv(
        root / 'manifests' / 'sample_identity.csv',
        dtype={'corpus_sentence_id': str},
    )
    assert not manifest.duplicated(['pair', 'corpus_sentence_id']).any()

    frames = {}
    pairs = None

    for mode in MODES:
        path = root / f'{mode}_all_stats.csv'
        df = pd.read_csv(path, dtype={'corpus_sentence_id': str})

        required = {
            'pair', 'sentence_id', 'corpus_sentence_id',
            'mean', 'src_len', 'tgt_len', 'analysis_view'
        }
        assert required.issubset(df.columns), (
            f'{mode}: missing collector columns {sorted(required - set(df.columns))}'
        )
        assert not df.duplicated(['pair', 'corpus_sentence_id']).any()
        assert len(df) == 40 * cfg['expected_n_sentences']
        assert df.pair.nunique() == 40
        assert df.groupby('pair').sentence_id.apply(
            lambda s: set(pd.to_numeric(s).astype(int)) == set(range(cfg['expected_n_sentences']))
        ).all()

        mode_pairs = sorted(df.pair.unique())
        if pairs is None:
            pairs = mode_pairs
        else:
            assert mode_pairs == pairs

        reference_ids = set(
            df.loc[df.pair == pairs[0], 'corpus_sentence_id'].astype(str)
        )
        assert all(
            set(g.corpus_sentence_id.astype(str)) == reference_ids
            for _, g in df.groupby('pair')
        )

        view = df['analysis_view'].astype(str)
        allowed_views = {
            'lexical_no_punctuation',
            'lexical_no_punctuation_unavailable',
        }
        if cfg['nllb_missingness'] and mode == 'free_generation':
            # The NLLB GPU collector labels the 57 declared capped generations
            # explicitly. They are retained for audit/quality only and are not
            # eligible attention observations.
            allowed_views.add('excluded_generation_cap')

        unexpected_views = sorted(set(view.unique()) - allowed_views)
        assert not unexpected_views, (
            f'{mode}: unexpected analysis_view value(s): {unexpected_views}'
        )

        if 'src_lexical_tokens' in df.columns:
            a = pd.to_numeric(df['src_len'], errors='coerce')
            b = pd.to_numeric(df['src_lexical_tokens'], errors='coerce')
            ok = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[ok], b[ok])
        if 'tgt_lexical_tokens' in df.columns:
            a = pd.to_numeric(df['tgt_len'], errors='coerce')
            b = pd.to_numeric(df['tgt_lexical_tokens'], errors='coerce')
            ok = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[ok], b[ok])

        if cfg['nllb_missingness'] and mode == 'free_generation':
            eligibility_columns = [
                'generation_hit_max_length',
                'attention_inference_eligible',
                'attention_tensor_role',
            ]
            assert all(c in df.columns for c in eligibility_columns)

            capped = _truthy(df['generation_hit_max_length'])
            eligible = (
                _truthy(df['attention_inference_eligible'])
                & ~capped
                & df['attention_tensor_role'].astype(str).str.lower().eq('inferential')
            )

            assert int(capped.sum()) == 57, (
                f'Expected 57 capped NLLB directions, found {int(capped.sum())}'
            )
            assert int(eligible.sum()) == 29943, (
                f'Expected 29,943 eligible NLLB directions, found {int(eligible.sum())}'
            )

            df['analysis_eligible'] = eligible
            df['exclusion_reason'] = np.where(
                capped, 'generation_capped',
                np.where(~eligible, 'metadata_ineligible', '')
            )
        else:
            df['analysis_eligible'] = True
            df['exclusion_reason'] = ''

        mean = pd.to_numeric(df['mean'], errors='coerce').to_numpy(float)
        src_len = pd.to_numeric(df['src_len'], errors='coerce').to_numpy(float)

        lexical_mass = mean * src_len
        eligible = df['analysis_eligible'].to_numpy(bool)
        lexical_mass = np.where(eligible, lexical_mass, np.nan)

        df['lexical_mass'] = lexical_mass
        df['src_len'] = pd.to_numeric(df['src_len'], errors='coerce')
        df['tgt_len'] = pd.to_numeric(df['tgt_len'], errors='coerce')

        unavailable = df['analysis_view'].astype(str).eq(
            'lexical_no_punctuation_unavailable'
        )
        df.loc[unavailable, 'lexical_mass'] = np.nan

        frames[mode] = df

    assert set(zip(frames['teacher_forced'].pair,
                   frames['teacher_forced'].corpus_sentence_id)) == set(
        zip(frames['free_generation'].pair,
            frames['free_generation'].corpus_sentence_id)
    )

    dyads = [
        (p, reverse_pair(p))
        for p in pairs
        if p < reverse_pair(p)
    ]
    assert len(dyads) == 20
    assert all(q in pairs for _, q in dyads)

    identity = manifest.set_index(['pair', 'corpus_sentence_id'])
    for p, q in dyads:
        a = identity.loc[p].sort_index()
        b = identity.loc[q].sort_index()
        assert a.index.equals(b.index), f'Reciprocal ID mismatch: {p}'
        assert (a.source_sha256.values == b.target_sha256.values).all(), p
        assert (a.target_sha256.values == b.source_sha256.values).all(), p

    ids = sorted(
        frames['teacher_forced'].corpus_sentence_id.astype(str).unique(),
        key=lambda s: (len(s), s)
    )
    assert len(ids) == cfg['expected_n_sentences']

    out.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        str(p.relative_to(root)): digest_file(p)
        for p in required_paths
    }

    provenance = {
        'dataset_label': cfg['dataset_label'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': str(root),
        'output_root': str(out),
        'input_hashes': input_hashes,
        'input_run_config': run_config,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'metric': 'lexical_mass = src_len * mean from GPU lexical-no-punctuation summary',
        'modes': MODES,
        'meaning_count': len(ids),
        'dyad_count': len(dyads),
        'nllb_missingness_policy': (
            'Free-generation attention is eligible only when '
            'attention_inference_eligible is true, generation did not hit max length, '
            'and attention_tensor_role is inferential. Reciprocal contrasts require '
            'both directions; missing/ineligible values remain NaN, never zero.'
            if cfg['nllb_missingness']
            else 'Reciprocal contrasts require finite lexical mass in both directions; '
                 'unavailable lexical values remain NaN, never zero.'
        ),
    }
    (out / 'provenance.json').write_text(
        json.dumps(provenance, indent=2, default=str),
        encoding='utf-8'
    )

    return {
        'root': root,
        'out': out,
        'frames': frames,
        'dyads': dyads,
        'ids': ids,
        'cfg': cfg,
    }


CTX = load_collection_context(CONFIG)
print(
    f"{CONFIG['dataset_label']}: "
    f"{len(CTX['dyads'])} reciprocal dyads, "
    f"{len(CTX['ids'])} shared meanings"
)
print('Outputs:', CTX['out'])


# %% original code cell 4; notebook cell 9
def potential_edge_data(ctx, mode):
    """Return meaning x dyad reciprocal lexical-mass asymmetry and length-balance mask."""
    indexed = ctx['frames'][mode].set_index(
        ['pair', 'corpus_sentence_id'],
        drop=False
    )

    edge_cols = []
    balanced_cols = []
    support_rows = []

    for p, q in ctx['dyads']:
        gp = indexed.loc[p].reindex(ctx['ids'])
        gq = indexed.loc[q].reindex(ctx['ids'])

        assert gp.index.astype(str).tolist() == ctx['ids']
        assert gq.index.astype(str).tolist() == ctx['ids']

        lm0 = pd.to_numeric(gp['lexical_mass'], errors='coerce').to_numpy(float)
        lm1 = pd.to_numeric(gq['lexical_mass'], errors='coerce').to_numpy(float)

        e0 = gp['analysis_eligible'].to_numpy(bool)
        e1 = gq['analysis_eligible'].to_numpy(bool)

        complete = (
            e0 & e1
            & np.isfinite(lm0)
            & np.isfinite(lm1)
        )

        d = np.full(len(ctx['ids']), np.nan, dtype=float)
        d[complete] = lm0[complete] - lm1[complete]

        s0 = pd.to_numeric(gp['src_len'], errors='coerce').to_numpy(float)
        s1 = pd.to_numeric(gq['src_len'], errors='coerce').to_numpy(float)
        t0 = pd.to_numeric(gp['tgt_len'], errors='coerce').to_numpy(float)
        t1 = pd.to_numeric(gq['tgt_len'], errors='coerce').to_numpy(float)

        length_ok = (
            np.isfinite(s0) & np.isfinite(s1)
            & np.isfinite(t0) & np.isfinite(t1)
            & (s0 > 0) & (s1 > 0)
            & (t0 > 0) & (t1 > 0)
        )

        src_log_ratio = np.full(len(d), np.nan)
        tgt_log_ratio = np.full(len(d), np.nan)
        src_log_ratio[length_ok] = np.log(s0[length_ok] / s1[length_ok])
        tgt_log_ratio[length_ok] = np.log(t0[length_ok] / t1[length_ok])

        balanced = (
            complete & length_ok
            & (np.abs(src_log_ratio) < 0.15)
            & (np.abs(tgt_log_ratio) < 0.15)
        )

        edge_cols.append(d)
        balanced_cols.append(balanced)
        support_rows.append({
            'mode': mode,
            'dyad': p,
            'n_primary': int(np.isfinite(d).sum()),
            'n_length_balanced': int(balanced.sum()),
            'n_excluded_reciprocal': int((~complete).sum()),
        })

    return (
        np.stack(edge_cols, axis=1),
        np.stack(balanced_cols, axis=1),
        pd.DataFrame(support_rows),
    )


EDGE_DATA = {}
BALANCED_MASK = {}
SUPPORT = []

for mode in MODES:
    e, b, s = potential_edge_data(CTX, mode)
    EDGE_DATA[mode] = e
    BALANCED_MASK[mode] = b
    SUPPORT.append(s)

SUPPORT = pd.concat(SUPPORT, ignore_index=True)
SUPPORT.to_csv(CTX['out'] / 'reciprocal_support.csv', index=False)

print('Reciprocal support')
display(
    SUPPORT.groupby('mode')[['n_primary', 'n_length_balanced',
                             'n_excluded_reciprocal']].agg(['min', 'max', 'sum'])
)


# %% original code cell 5; notebook cell 11
def graph_decomposition(ctx, edge_data):
    """Base conservative-field decomposition using the same graph and meaning-fold definition."""
    langs, inc = incidence_matrix(ctx['dyads'])
    pinv = np.linalg.pinv(inc)

    fold = np.array([
        int(hashlib.sha256(str(i).encode()).hexdigest()[:8], 16)
        % ctx['cfg']['folds']
        for i in ctx['ids']
    ])

    edge_rows = []
    node_rows = []
    summary_rows = []

    for mode in MODES:
        edge = edge_data[mode]
        means = safe_mean(edge, 0)
        assert np.isfinite(means).all(), f'{mode}: an observed graph edge has no finite support'

        theta = pinv @ means
        gradient = inc @ theta
        residual = means - gradient

        sse = 0.0
        base = 0.0
        scored_folds = 0

        for k in np.unique(fold):
            train = fold != k
            test = fold == k
            if not train.any() or not test.any():
                continue

            train_means = safe_mean(edge[train], 0)
            test_means = safe_mean(edge[test], 0)
            assert np.isfinite(train_means).all()
            assert np.isfinite(test_means).all()

            pred = inc @ (pinv @ train_means)
            sse += float(np.sum((test_means - pred) ** 2))
            base += float(np.sum(test_means ** 2))
            scored_folds += 1

        energy_denom = float(np.sum(means ** 2))
        energy = (
            float(np.sum(gradient ** 2) / energy_denom)
            if energy_denom > 0 else np.nan
        )
        oof_r2 = float(1 - sse / base) if base > 0 else np.nan
        rank = int(np.linalg.matrix_rank(inc))

        for j, (p, q) in enumerate(ctx['dyads']):
            edge_rows.append({
                'mode': mode,
                'metric': 'lexical_mass',
                'dyad': p,
                'observed': float(means[j]),
                'language_gradient': float(gradient[j]),
                'cycle_residual': float(residual[j]),
                'graph_rank': rank,
                'in_sample_gradient_energy_fraction': energy,
                'out_of_fold_R2_against_zero': oof_r2,
            })

        for j, lang in enumerate(langs):
            node_rows.append({
                'mode': mode,
                'metric': 'lexical_mass',
                'language': lang,
                'source_minus_target_role_potential': float(theta[j]),
            })

        summary_rows.append({
            'mode': mode,
            'metric': 'lexical_mass',
            'n_edges': len(ctx['dyads']),
            'n_languages': len(langs),
            'graph_rank': rank,
            'gradient_energy_fraction': energy,
            'meaning_OOF_R2_against_zero': oof_r2,
            'scored_meaning_folds': scored_folds,
            'total_meaning_edge_observations': int(np.isfinite(edge).sum()),
        })

    edges = pd.DataFrame(edge_rows)
    nodes = pd.DataFrame(node_rows)
    summary = pd.DataFrame(summary_rows)

    edges.to_csv(ctx['out'] / 'language_graph_decomposition.csv', index=False)
    nodes.to_csv(ctx['out'] / 'language_role_potentials.csv', index=False)
    summary.to_csv(ctx['out'] / 'potential_field_base_summary.csv', index=False)

    return edges, nodes, summary


GRAPH_EDGES, LANGUAGE_POTENTIALS, BASE_SUMMARY = graph_decomposition(
    CTX, EDGE_DATA
)

print('Base conservative-field result')
display(BASE_SUMMARY)
display(
    GRAPH_EDGES[['mode', 'dyad', 'observed',
                 'language_gradient', 'cycle_residual']]
)
print('Language potentials')
display(LANGUAGE_POTENTIALS)


# %% original code cell 6; notebook cell 13
def edge_holdout_validation(ctx, edge_data):
    """Leave one dyad edge out; fit theta on all other full-rank edges."""
    langs, inc = incidence_matrix(ctx['dyads'])
    full_rank = int(np.linalg.matrix_rank(inc))
    rows, summaries = [], []

    for mode in MODES:
        edge = edge_data[mode]
        means = safe_mean(edge, 0)

        for j, (p, q) in enumerate(ctx['dyads']):
            train_edges = np.ones(len(ctx['dyads']), dtype=bool)
            train_edges[j] = False
            train_edges &= np.isfinite(means)

            train_rank = (
                int(np.linalg.matrix_rank(inc[train_edges]))
                if train_edges.any() else 0
            )
            scorable = bool(
                np.isfinite(means[j]) and train_rank == full_rank
            )

            pred = error = np.nan
            if scorable:
                theta = np.linalg.pinv(inc[train_edges]) @ means[train_edges]
                pred = float(inc[j] @ theta)
                error = float(pred - means[j])

            rows.append({
                'mode': mode,
                'metric': 'lexical_mass',
                'heldout_dyad': p,
                'actual': float(means[j]) if np.isfinite(means[j]) else np.nan,
                'predicted_from_other_dyads': pred,
                'signed_error': error,
                'absolute_error': abs(error) if np.isfinite(error) else np.nan,
                'train_edges': int(train_edges.sum()),
                'train_graph_rank': train_rank,
                'full_graph_rank': full_rank,
                'scorable': scorable,
                'note': (
                    '' if scorable else
                    'Held-out edge is not identifiable from a connected full-rank training graph.'
                ),
            })

        mode_rows = pd.DataFrame([r for r in rows if r['mode'] == mode])
        scored = (
            mode_rows.scorable
            & np.isfinite(mode_rows.actual)
            & np.isfinite(mode_rows.predicted_from_other_dyads)
        )
        y = mode_rows.loc[scored, 'actual'].to_numpy(float)
        pred = mode_rows.loc[
            scored, 'predicted_from_other_dyads'
        ].to_numpy(float)

        pearson = (
            float(np.corrcoef(y, pred)[0, 1])
            if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0
            else np.nan
        )
        spear = (
            float(spearmanr(y, pred).statistic)
            if len(y) >= 2 else np.nan
        )

        summaries.append({
            'mode': mode,
            'metric': 'lexical_mass',
            'n_scorable_edges': int(scored.sum()),
            'R2_against_zero': r2_against_zero(y, pred),
            'MAE': float(np.mean(np.abs(y - pred))) if len(y) else np.nan,
            'RMSE': (
                float(np.sqrt(np.mean((y - pred) ** 2)))
                if len(y) else np.nan
            ),
            'pearson_observed_predicted': pearson,
            'spearman_observed_predicted': spear,
            'full_graph_rank': full_rank,
        })

    return pd.DataFrame(rows), pd.DataFrame(summaries)


EDGE_HOLDOUT_PREDICTIONS, EDGE_HOLDOUT_SUMMARY = edge_holdout_validation(
    CTX, EDGE_DATA
)

EDGE_HOLDOUT_PREDICTIONS.to_csv(
    CTX['out'] / 'potential_edge_holdout_predictions.csv', index=False
)
EDGE_HOLDOUT_SUMMARY.to_csv(
    CTX['out'] / 'potential_edge_holdout_summary.csv', index=False
)

print('Leave-one-dyad-out prediction')
display(EDGE_HOLDOUT_SUMMARY)
display(
    EDGE_HOLDOUT_PREDICTIONS.sort_values(['mode', 'absolute_error'])
)


# %% original code cell 7; notebook cell 15
def length_balanced_potential_validation(ctx, edge_data, balanced_mask):
    """Graph decomposition on the frozen lexical-length-balanced meanings only."""
    langs, inc = incidence_matrix(ctx['dyads'])
    full_rank = int(np.linalg.matrix_rank(inc))

    fold = np.array([
        int(hashlib.sha256(str(i).encode()).hexdigest()[:8], 16)
        % ctx['cfg']['folds']
        for i in ctx['ids']
    ])

    rows, summaries = [], []

    for mode in MODES:
        edge = edge_data[mode]
        balanced = balanced_mask[mode]
        balanced_edge = np.where(balanced, edge, np.nan)

        counts = np.isfinite(balanced_edge).sum(0)
        means = safe_mean(balanced_edge, 0)
        fit_edges = (
            np.isfinite(means)
            & (counts >= ctx['cfg']['min_paired'])
        )
        fit_rank = (
            int(np.linalg.matrix_rank(inc[fit_edges]))
            if fit_edges.any() else 0
        )

        gradient = np.full(len(ctx['dyads']), np.nan)
        residual = np.full(len(ctx['dyads']), np.nan)
        energy = np.nan

        if fit_rank == full_rank:
            theta = np.linalg.pinv(inc[fit_edges]) @ means[fit_edges]
            gradient = inc @ theta
            residual = means - gradient
            denom = float(np.sum(means[fit_edges] ** 2))
            energy = (
                float(np.sum(gradient[fit_edges] ** 2) / denom)
                if denom > 0 else np.nan
            )

        sse = 0.0
        base = 0.0
        scored_folds = 0

        for k in np.unique(fold):
            train = fold != k
            test = fold == k

            y_train = safe_mean(balanced_edge[train], 0)
            n_train = np.isfinite(balanced_edge[train]).sum(0)
            train_edges = (
                np.isfinite(y_train)
                & (n_train >= ctx['cfg']['min_paired'])
            )

            if (
                (not train_edges.any())
                or int(np.linalg.matrix_rank(inc[train_edges])) != full_rank
            ):
                continue

            theta_k = (
                np.linalg.pinv(inc[train_edges])
                @ y_train[train_edges]
            )

            y_test = safe_mean(balanced_edge[test], 0)
            test_edges = np.isfinite(y_test)
            if not test_edges.any():
                continue

            pred = inc[test_edges] @ theta_k
            finite = (
                np.isfinite(y_test[test_edges])
                & np.isfinite(pred)
            )
            if not finite.any():
                continue

            y_eval = y_test[test_edges][finite]
            pred_eval = pred[finite]
            sse += float(np.sum((y_eval - pred_eval) ** 2))
            base += float(np.sum(y_eval ** 2))
            scored_folds += 1

        oof_r2 = float(1 - sse / base) if base > 0 else np.nan

        for j, (p, q) in enumerate(ctx['dyads']):
            rows.append({
                'mode': mode,
                'metric': 'lexical_mass',
                'dyad': p,
                'n_length_balanced': int(counts[j]),
                'length_balanced_fraction_of_primary': float(
                    counts[j] / max(1, np.isfinite(edge[:, j]).sum())
                ),
                'observed_length_balanced': (
                    float(means[j]) if np.isfinite(means[j]) else np.nan
                ),
                'language_gradient_length_balanced': (
                    float(gradient[j]) if np.isfinite(gradient[j]) else np.nan
                ),
                'cycle_residual_length_balanced': (
                    float(residual[j]) if np.isfinite(residual[j]) else np.nan
                ),
            })

        summaries.append({
            'mode': mode,
            'metric': 'lexical_mass',
            'fit_edges': int(fit_edges.sum()),
            'full_graph_rank': full_rank,
            'length_balanced_graph_rank': fit_rank,
            'gradient_energy_fraction_length_balanced': energy,
            'meaning_OOF_R2_against_zero_length_balanced': oof_r2,
            'scored_meaning_folds': int(scored_folds),
            'total_primary_meaning_edge_observations': int(
                np.isfinite(edge).sum()
            ),
            'total_length_balanced_meaning_edge_observations': int(
                np.isfinite(balanced_edge).sum()
            ),
        })

    return pd.DataFrame(rows), pd.DataFrame(summaries)


LENGTH_BALANCED_EDGES, LENGTH_BALANCED_SUMMARY = (
    length_balanced_potential_validation(
        CTX, EDGE_DATA, BALANCED_MASK
    )
)

LENGTH_BALANCED_EDGES.to_csv(
    CTX['out'] / 'potential_length_balanced_edges.csv', index=False
)
LENGTH_BALANCED_SUMMARY.to_csv(
    CTX['out'] / 'potential_length_balanced_summary.csv', index=False
)

print('Lexical-length-balanced sensitivity')
display(LENGTH_BALANCED_SUMMARY)
display(
    LENGTH_BALANCED_EDGES[
        ['mode', 'dyad', 'n_length_balanced',
         'observed_length_balanced',
         'language_gradient_length_balanced',
         'cycle_residual_length_balanced']
    ]
)


# %% original code cell 8; notebook cell 17
def random_edge_null_validation(ctx, edge_data):
    """Permute observed signed dyad means across the fixed graph edges."""
    langs, inc = incidence_matrix(ctx['dyads'])
    rows = []
    repeats = int(ctx['cfg']['permutations'])

    for mode_i, mode in enumerate(MODES):
        edge = edge_data[mode]
        means = safe_mean(edge, 0)
        valid = np.isfinite(means)

        A = inc[valid]
        y = means[valid]
        rank = int(np.linalg.matrix_rank(A)) if len(y) else 0
        pinv = np.linalg.pinv(A)

        gradient = A @ (pinv @ y)
        denom = float(np.sum(y ** 2))
        observed_energy = (
            float(np.sum(gradient ** 2) / denom)
            if denom > 0 else np.nan
        )

        rng = np.random.default_rng(
            ctx['cfg']['seed'] + 700 + mode_i
        )
        null = np.full(repeats, np.nan)

        if denom > 0:
            for b in range(repeats):
                yp = rng.permutation(y)
                gp = A @ (pinv @ yp)
                null[b] = float(
                    np.sum(gp ** 2) / np.sum(yp ** 2)
                )

        finite_null = null[np.isfinite(null)]
        p = (
            (
                (finite_null >= observed_energy - 1e-15).sum() + 1
            ) / (len(finite_null) + 1)
            if len(finite_null) else np.nan
        )

        rows.append({
            'mode': mode,
            'metric': 'lexical_mass',
            'n_edges': int(valid.sum()),
            'graph_rank': rank,
            'observed_gradient_energy_fraction': observed_energy,
            'null_repeats': int(len(finite_null)),
            'null_mean': (
                float(np.mean(finite_null))
                if len(finite_null) else np.nan
            ),
            'null_q95': (
                float(np.quantile(finite_null, 0.95))
                if len(finite_null) else np.nan
            ),
            'null_q99': (
                float(np.quantile(finite_null, 0.99))
                if len(finite_null) else np.nan
            ),
            'null_max': (
                float(np.max(finite_null))
                if len(finite_null) else np.nan
            ),
            'permutation_p': (
                float(p) if np.isfinite(p) else np.nan
            ),
            'null_definition': (
                'Permute observed signed dyad means across the fixed graph edges.'
            ),
        })

    return pd.DataFrame(rows)


RANDOM_EDGE_NULL_SUMMARY = random_edge_null_validation(
    CTX, EDGE_DATA
)
RANDOM_EDGE_NULL_SUMMARY.to_csv(
    CTX['out'] / 'potential_random_edge_null_summary.csv',
    index=False
)

print('Random-edge permutation null')
display(RANDOM_EDGE_NULL_SUMMARY)


# %% original code cell 9; notebook cell 19
FINAL_SUMMARY = (
    BASE_SUMMARY
    .merge(
        EDGE_HOLDOUT_SUMMARY,
        on=['mode', 'metric'],
        how='left',
        suffixes=('', '_loeo')
    )
    .merge(
        LENGTH_BALANCED_SUMMARY,
        on=['mode', 'metric'],
        how='left'
    )
    .merge(
        RANDOM_EDGE_NULL_SUMMARY,
        on=['mode', 'metric'],
        how='left',
        suffixes=('', '_null')
    )
)

keep = [
    'mode', 'metric',
    'n_edges', 'n_languages', 'graph_rank',
    'gradient_energy_fraction',
    'meaning_OOF_R2_against_zero',
    'total_meaning_edge_observations',
    'n_scorable_edges',
    'R2_against_zero',
    'MAE', 'RMSE',
    'pearson_observed_predicted',
    'spearman_observed_predicted',
    'fit_edges',
    'length_balanced_graph_rank',
    'gradient_energy_fraction_length_balanced',
    'meaning_OOF_R2_against_zero_length_balanced',
    'total_primary_meaning_edge_observations',
    'total_length_balanced_meaning_edge_observations',
    'observed_gradient_energy_fraction',
    'null_repeats', 'null_mean', 'null_q95', 'null_q99', 'null_max',
    'permutation_p',
]
keep = [c for c in keep if c in FINAL_SUMMARY.columns]
FINAL_SUMMARY = FINAL_SUMMARY[keep]

FINAL_SUMMARY.to_csv(
    CTX['out'] / 'conservative_field_final_summary.csv',
    index=False
)

analysis_config = {
    'metric': 'lexical_mass',
    'lexical_mass_reconstruction':
        'lexical_mass = src_len * mean from the GPU collector primary lexical-no-punctuation map',
    'folds': CONFIG['folds'],
    'seed': CONFIG['seed'],
    'permutations': CONFIG['permutations'],
    'min_paired': CONFIG['min_paired'],
    'length_balance_rule':
        'abs(log(source_lexical_length_forward/source_lexical_length_reverse)) < 0.15 '
        'and abs(log(target_lexical_length_forward/target_lexical_length_reverse)) < 0.15',
    'loeo_scope':
        'Fixed observed language set; one observed dyad edge held out at a time.',
    'random_edge_null':
        'Permute observed signed dyad means across the fixed observed graph edges.',
    'missingness_rule':
        'Reciprocal meaning-edge values require both directions and finite lexical mass; '
        'missing/ineligible observations are NaN, never zero.',
}
(CTX['out'] / 'analysis_config.json').write_text(
    json.dumps(analysis_config, indent=2),
    encoding='utf-8'
)

print('FINAL CONSERVATIVE-FIELD SUMMARY')
display(FINAL_SUMMARY)

print('\nSaved canonical outputs to:')
print(CTX['out'])

