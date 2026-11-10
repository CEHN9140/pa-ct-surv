"""Read-only, patient-level diagnostics of an existing five-fold survival run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pact_matplotlib')

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_features(path, input_dim=1024, block_size=512):
    with h5py.File(path, 'r') as handle:
        features = handle['features']
        if features.ndim != 2 or features.shape[1] != input_dim or not len(features):
            raise ValueError(f'Invalid feature shape: {path}: {features.shape}')
        total = np.zeros(input_dim, dtype=np.float64)
        norm_sum = norm_square_sum = 0.
        zeros = 0
        for start in range(0, len(features), block_size):
            block = np.asarray(features[start:start+block_size], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError(f'Non-finite features: {path}')
            norms = np.linalg.norm(block, axis=1)
            total += block.sum(0)
            norm_sum += norms.sum()
            norm_square_sum += np.square(norms).sum()
            zeros += int((norms == 0).sum())
        count = len(features)
    norm_mean = norm_sum/count
    return total/count, dict(n_patches=count, zero_patches=zeros,
                            patch_norm_mean=norm_mean,
                            patch_norm_sd=np.sqrt(max(0., norm_square_sum/count-norm_mean**2)))


def load_feature_summaries(data, cache_path, csv_hash):
    fingerprint = [(str(Path(p).resolve()), Path(p).stat().st_size,
                    Path(p).stat().st_mtime_ns) for p in data.pa_path]
    signature = hashlib.sha256(json.dumps([csv_hash, fingerprint]).encode()).hexdigest()
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if cached['signature'].item() == signature:
                print('Using verified all-patch feature summary cache.', flush=True)
                return cached['means'].copy(), pd.DataFrame(
                    cached['statistics'], columns=cached['columns'].tolist()), signature
    means, statistics = [], []
    for index, path in enumerate(data.pa_path):
        mean, stats = summarize_features(path)
        means.append(mean)
        statistics.append(stats)
        if (index+1) % 25 == 0 or index+1 == len(data):
            print(f'Summarized ALL patches: {index+1}/{len(data)} patients', flush=True)
    current = [(str(Path(p).resolve()), Path(p).stat().st_size,
                Path(p).stat().st_mtime_ns) for p in data.pa_path]
    if current != fingerprint:
        raise RuntimeError('Input feature files changed during reading')
    means, statistics = np.stack(means), pd.DataFrame(statistics)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, signature=signature, means=means,
                        statistics=statistics.to_numpy(), columns=statistics.columns.to_numpy(dtype=str))
    return means, statistics, signature


def read_predictions(path, patients):
    predictions = pd.read_csv(path, dtype={'case_id': str})
    if predictions.case_id.duplicated().any() or set(predictions.case_id) != set(patients.pa_id):
        raise ValueError(f'Prediction patients do not match the exact partition: {path}')
    predictions = predictions.set_index('case_id').loc[patients.pa_id].reset_index()
    values = predictions[['dfs.month', 'dfs.status', 'risk_score']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite prediction values: {path}')
    # Training stores DFS times as float32; compare at that exact representation.
    if not (np.array_equal(values[:, 0].astype(np.float32), patients.time.to_numpy(np.float32))
            and np.array_equal(values[:, 1], patients.event.to_numpy())):
        raise ValueError(f'Prediction labels do not match the source CSV: {path}')
    return predictions


def pair_analysis(frame):
    time = frame['dfs.month'].to_numpy(float)
    event = frame['dfs.status'].to_numpy(bool)
    risk = frame.risk_score.to_numpy(float)
    comparable = event[:, None] & ((time[:, None] < time[None, :]) |
                                  ((time[:, None] == time[None, :]) & ~event[None, :]))
    difference = risk[:, None]-risk[None, :]
    correct = (difference > 1e-8).astype(float)+.5*(np.abs(difference) <= 1e-8)
    count = comparable.sum(0)+comparable.sum(1)
    wrong = (1-correct)*comparable
    error_rate = np.divide(wrong.sum(0)+wrong.sum(1), count,
                           out=np.full(len(time), np.nan), where=count > 0)
    summary = dict(cindex=float(correct[comparable].mean()), n_pairs=int(comparable.sum()))
    errors = pd.DataFrame(dict(pair_count=count, error_rate=error_rate))
    bins = np.digitize(time, [12, 24, 36], right=True)
    strata = []
    for group in range(4):
        for partner_event in [0, 1]:
            selected = comparable & (bins == group)[:, None] & (event[None, :] == partner_event)
            n = int(selected.sum())
            strata.append(dict(time_bin=group, partner_event=partner_event, n_pairs=n,
                               correct_sum=float(correct[selected].sum()),
                               cindex=float(correct[selected].mean()) if n else np.nan))
    return summary, errors, pd.DataFrame(strata)


def origin_embeddings(x, event, seed, pca_dim):
    """Splits depend on event, never on observed/permuted origin labels."""
    result = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, event):
        scaler = StandardScaler().fit(x[train])
        train_x = scaler.transform(x[train])
        dim = min(pca_dim, train_x.shape[1], len(train)-1)
        pca = PCA(dim, svd_solver='randomized', random_state=seed).fit(train_x)
        result.append((train, test, pca.transform(train_x), pca.transform(scaler.transform(x[test]))))
    return result


def permute_within_events(origin, event, n_permutations, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.tile(origin, (n_permutations, 1))
    for labels in shuffled:
        for value in np.unique(event):
            indices = np.flatnonzero(event == value)
            labels[indices] = rng.permutation(labels[indices])
    return shuffled


def origin_test(x, origin, event, seed=123, n_permutations=999, pca_dim=32):
    embeddings = origin_embeddings(x, event, seed, pca_dim)

    def predict(labels):
        probabilities = np.full(len(labels), np.nan)
        aucs = []
        for train, test, train_x, test_x in embeddings:
            classifier = LogisticRegression(C=1., class_weight='balanced', max_iter=2000)
            classifier.fit(train_x, labels[train])
            probabilities[test] = classifier.predict_proba(test_x)[:, 1]
            aucs.append(roc_auc_score(labels[test], probabilities[test]))
        return np.mean(aucs), np.std(aucs, ddof=1), probabilities

    auc, auc_sd, probabilities = predict(origin)
    null = np.array([predict(labels)[0] for labels in
                     permute_within_events(origin, event, n_permutations, seed)])
    result = dict(origin_auc=float(auc), origin_auc_inner_sd=float(auc_sd),
                  origin_p=float((1+(null >= auc).sum())/(1+len(null))),
                  origin_null025=float(np.quantile(null, .025)),
                  origin_null975=float(np.quantile(null, .975)))
    return result, probabilities, null


def training_distances(train, val, seed=123, pca_dim=32):
    scaler = StandardScaler().fit(train)
    train_x = scaler.transform(train)
    pca = PCA(min(pca_dim, train.shape[1], len(train)-1), whiten=True,
              svd_solver='randomized', random_state=seed).fit(train_x)
    train_x = pca.transform(train_x)
    neighbors = NearestNeighbors(n_neighbors=min(5, len(train))).fit(train_x)
    return neighbors.kneighbors(pca.transform(scaler.transform(val)))[0].mean(1)


def holm_adjust(values):
    values = np.asarray(values)
    order = np.argsort(values)
    corrected = np.empty_like(values)
    corrected[order] = np.minimum(1., np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return corrected


def error_associations(patients):
    rows = []
    time_bins = np.digitize(patients.time.to_numpy(), [12, 24, 36], right=True)
    strata = patients.event.to_numpy()*4+time_bins
    covariates = np.column_stack([np.ones(len(patients)), pd.get_dummies(strata, drop_first=True).to_numpy(float)])
    for measure in ['train_distance', 'origin_probability', 'log_patches', 'patch_norm_mean', 'patch_norm_sd']:
        valid = np.isfinite(patients[measure]) & np.isfinite(patients.error_rate)
        x, error = patients.loc[valid, measure].to_numpy(), patients.loc[valid, 'error_rate'].to_numpy()
        rho = float(spearmanr(x, error).statistic) if np.ptp(x) and np.ptp(error) else np.nan
        # Partial rank correlation is descriptive; shared patient pairs preclude naive independent p-values.
        design = covariates[valid]
        residuals = []
        for values in [x, error]:
            ranks = rankdata(values)
            residuals.append(ranks-design@np.linalg.lstsq(design, ranks, rcond=None)[0])
        adjusted = float(np.corrcoef(residuals)[0, 1]) if all(np.std(r) > 1e-12 for r in residuals) else np.nan
        far = x > np.median(x)
        rows.append(dict(measure=measure, spearman=rho, time_event_adjusted_rank_correlation=adjusted,
                         near_n=int((~far).sum()), far_n=int(far.sum()),
                         near_mean_patient_error=float(error[~far].mean()),
                         far_mean_patient_error=float(error[far].mean()) if far.any() else np.nan))
    return pd.DataFrame(rows)


def plot_overview(summary, associations, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout='constrained')
    fold = summary.fold.to_numpy()
    axes[0].plot(fold, summary.train_cindex, 'o-', color='#4C78A8', lw=3, label='Train')
    axes[0].plot(fold, summary.val_cindex, 'o-', color='#E19C55', lw=3, label='Validation')
    axes[0].set(title='A  Existing ABMIL rankings', ylabel='C-index', ylim=(.5, 1))
    axes[0].legend(frameon=False)
    axes[1].vlines(fold, summary.origin_null025, summary.origin_null975, color='0.65', lw=8,
                   label='95% permutation null range')
    axes[1].scatter(fold, summary.origin_auc, s=75, color='#4C78A8', zorder=3, label='Held-out origin AUC')
    axes[1].axhline(.5, color='0.4', ls='--')
    axes[1].set(title='B  Input distribution probe', ylabel='Origin classifier AUC')
    axes[1].legend(frameon=False, fontsize=10)
    for measure, label, color in [('train_distance', 'Distance to training patients', '#4C78A8'),
                                   ('origin_probability', 'Validation-origin probability', '#E19C55')]:
        part = associations.loc[associations.measure == measure]
        axes[2].plot(part.fold, part.spearman, 'o-', lw=3, color=color, label=label)
    axes[2].axhline(0, color='0.4', ls='--')
    axes[2].set(title='C  Association with ranking errors', ylabel='Descriptive Spearman correlation', ylim=(-1, 1))
    axes[2].legend(frameon=False, fontsize=10)
    for ax in axes:
        ax.set(xlabel='Survival validation fold', xticks=fold)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'diagnostic_overview.png', dpi=220)
    fig.savefig(output/'diagnostic_overview.pdf')
    plt.close(fig)


def analyze(args):
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data_csv, dtype={'pa_id': str, 'ct_id': str})
    if (data.pa_id.duplicated().any() or data.ct_id.duplicated().any() or data.isna().any().any()
            or set(data.split) != set(range(5)) or not data.event.isin([0, 1]).all()
            or not np.isfinite(data.time).all() or (data.time < 0).any()):
        raise ValueError('Invalid patient identifiers, labels, paths or five-fold assignments')
    # Validate all predictions before expensive input reads or diagnostic fitting.
    predictions, provenance = {}, {}
    for fold in range(5):
        for part in ['train', 'val']:
            mask = data.split == fold if part == 'val' else data.split != fold
            path = Path(args.prediction_root)/f'fold_{fold}'/f'{part}_predictions.csv'
            predictions[fold, part] = read_predictions(path, data.loc[mask])
            provenance[str(path.resolve())] = file_hash(path)
    csv_hash = file_hash(args.data_csv)
    x, statistics, signature = load_feature_summaries(data, output.parent/'cache'/'patient_features.npz', csv_hash)
    for column in statistics:
        data[column] = statistics[column].to_numpy()
    data['log_patches'] = np.log1p(data.n_patches)
    metadata_path = Path(args.metadata_csv)
    if metadata_path.is_file():
        old = pd.read_csv(metadata_path, dtype={'pa_id': str, 'ct_id': str})
        if old.pa_id.duplicated().any() or set(old.pa_id) != set(data.pa_id):
            raise ValueError('Source-batch metadata patients differ from current cohort')
        old = old.set_index('pa_id').loc[data.pa_id]
        if not (np.array_equal(old.ct_id, data.ct_id) and
                np.array_equal(old.event, data.event) and np.array_equal(old.split, data.split)
                and np.allclose(old.time, data.time, atol=1e-6, rtol=0)):
            raise ValueError('Source-batch metadata labels or splits differ from current cohort')
        data['source_batch'] = old.source_batch.to_numpy()
        provenance[str(metadata_path.resolve())] = file_hash(metadata_path)
    data.to_csv(output/'patient_summaries.csv', index=False)
    summaries, distribution_rows, patients, breakdown, all_associations, permutations = [], [], [], [], [], []
    source_rows = []
    for fold in range(5):
        print(f'Fold {fold}: origin classifier and {args.permutations} event-stratified permutations...', flush=True)
        val = (data.split == fold).to_numpy()
        result, origin_probability, null = origin_test(x, val.astype(int), data.event.to_numpy(),
                                                      args.seed, args.permutations, args.pca_dim)
        permutations.append(pd.DataFrame({'fold': fold, 'permutation': np.arange(1, len(null)+1), 'auc': null}))
        distances = training_distances(x[~val], x[val], args.seed, args.pca_dim)
        row = dict(fold=fold, **result)
        strata_parts = []
        for part, mask in [('train', ~val), ('val', val)]:
            cohort = data.loc[mask].reset_index(drop=True)
            metrics, errors, strata = pair_analysis(predictions[fold, part])
            row[f'{part}_cindex'] = metrics['cindex']
            row[f'{part}_n'] = len(cohort)
            cohort['risk_score'] = predictions[fold, part].risk_score.to_numpy()
            cohort = pd.concat([cohort, errors], axis=1)
            cohort['fold'], cohort['part'] = fold, part
            cohort['origin_probability'] = origin_probability[mask]
            if part == 'val':
                cohort['train_distance'] = distances
                association = error_associations(cohort)
                association['fold'] = fold
                all_associations.append(association)
                for measure, name in [('train_distance', 'distance_error_rho'),
                                      ('origin_probability', 'origin_error_rho')]:
                    row[name] = association.loc[association.measure == measure, 'spearman'].iloc[0]
            patients.append(cohort)
            stats = dict(fold=fold, part=part, n=len(cohort), events=int(cohort.event.sum()),
                         event_rate=float(cohort.event.mean()),
                         event_time_median=float(cohort.loc[cohort.event == 1, 'time'].median()),
                         censor_time_median=float(cohort.loc[cohort.event == 0, 'time'].median()))
            for measure in ['n_patches', 'patch_norm_mean', 'patch_norm_sd']:
                stats[f'{measure}_mean'] = float(cohort[measure].mean())
                stats[f'{measure}_median'] = float(cohort[measure].median())
            distribution_rows.append(stats)
            if 'source_batch' in cohort:
                for source, group in cohort.groupby('source_batch'):
                    source_rows.append(dict(fold=fold, part=part, source_batch=source, n=len(group),
                                            proportion=len(group)/len(cohort),
                                            mean_patient_error=group.error_rate.mean()))
            strata['fold'], strata['part'] = fold, part
            breakdown.append(strata)
            strata_parts.append(strata)
        counts = np.array([part.n_pairs.to_numpy() for part in strata_parts])
        hits = np.array([part.correct_sum.to_numpy() for part in strata_parts])
        common = (counts > 0).all(0)
        weights = counts[:, common].sum(0)/counts[:, common].sum()
        standardized = (hits[:, common]/counts[:, common]*weights).sum(1)
        row['standardized_pair_mass_coverage'] = float(counts[:, common].sum()/counts.sum())
        row['gap'] = row['train_cindex']-row['val_cindex']
        row['standardized_train_cindex'], row['standardized_val_cindex'] = standardized
        row['standardized_gap'] = standardized[0]-standardized[1]
        summaries.append(row)
        print(f"Fold {fold}: Train={row['train_cindex']:.4f}, Val={row['val_cindex']:.4f}, "
              f"origin AUC={row['origin_auc']:.4f}, permutation p={row['origin_p']:.4f}, "
              f"distance/error rho={row['distance_error_rho']:.4f}", flush=True)
    summary = pd.DataFrame(summaries)
    summary['origin_p_holm'] = holm_adjust(summary.origin_p)
    associations = pd.concat(all_associations, ignore_index=True)
    summary.to_csv(output/'fold_summary.csv', index=False)
    pd.DataFrame(distribution_rows).to_csv(output/'cohort_distributions.csv', index=False)
    pd.concat(patients, ignore_index=True).to_csv(output/'patient_errors.csv', index=False)
    pd.concat(breakdown, ignore_index=True).to_csv(output/'pair_breakdown.csv', index=False)
    associations.to_csv(output/'error_associations.csv', index=False)
    pd.concat(permutations, ignore_index=True).to_csv(output/'origin_permutations.csv', index=False)
    if source_rows:
        pd.DataFrame(source_rows).to_csv(output/'source_batch_summary.csv', index=False)
    provenance[str(Path(args.data_csv).resolve())] = csv_hash
    manifest = dict(arguments=vars(args), input_hashes=provenance, feature_file_signature=signature,
                    n_patients=len(data), n_patches=int(data.n_patches.sum()),
                    classifier='StandardScaler / PCA32(no whitening) / LogisticRegression(C=1, balanced)',
                    distance='Train-only StandardScaler / whitened PCA / mean distance to 5 nearest training patients',
                    association='Descriptive; no independent-pair p-values or causal interpretation')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    plot_overview(summary, associations, output)
    lines = ['# 实验07结果', '',
             f'患者{len(data)}名，event=1共{int(data.event.sum())}名；全部patch共{int(data.n_patches.sum()):,}个。', '',
             '每折均核对训练/验证预测病例与标签；以下Train/Val为同一最佳checkpoint的已有评估预测。', '',
             '|Fold|Train CI|Val CI|Gap|统一患者对组成后Gap|来源AUC|置换P|Holm P|距离/错误相关|',
             '|---|---|---|---|---|---|---|---|---|']
    for r in summary.itertuples():
        lines.append(f'|{r.fold}|{r.train_cindex:.4f}|{r.val_cindex:.4f}|{r.gap:.4f}|'
                     f'{r.standardized_gap:.4f}|{r.origin_auc:.4f}|{r.origin_p:.4f}|'
                     f'{r.origin_p_holm:.4f}|{r.distance_error_rho:.4f}|')
    lines += ['', '来源AUC使用内部未见患者；灰色区间为置换零假设区间，不是置信区间。',
              '来源检验针对平均UNI特征的简化表示。未检出不证明完全同分布；检出不证明导致泛化差距。',
              '相关系数、远近组错误率及时间/event调整后秩相关均为描述性分析。患者对并非独立。',
              '统一患者对组成后的CI只用于诊断，不替代正式CI；具体分组见pair_breakdown.csv。',
              '全部分析条件于已训练、已选模的预测，不包含重新训练及选模不确定性。', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(summary[['fold', 'train_cindex', 'val_cindex', 'origin_auc', 'origin_p_holm',
                   'distance_error_rho', 'origin_error_rho']].to_string(index=False), flush=True)
    print(f'Outputs: {output.resolve()}', flush=True)


def main():
    here = Path(__file__).resolve().parent
    base = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_csv', default=str(base/'data/trident/features_uni_v1/all_label_roi64.csv'))
    parser.add_argument('--prediction_root', default=str(base/'results/pact_v6/pathology/'
                        'trident_features_uni_v1_cv5_roi64_abmil_cox_dropout0.25_epochs30_coxbs64_lr1e-4_wd5e-4'))
    parser.add_argument('--metadata_csv', default=str(base/'tests/fold_difficulty_analysis/cohort.csv'))
    parser.add_argument('--output_root', default=str(here/'results'))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--pca_dim', type=int, default=32)
    args = parser.parse_args()
    if args.permutations < 1 or args.pca_dim < 1:
        parser.error('permutations and pca_dim must be positive')
    warnings.filterwarnings('error', category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        analyze(args)


if __name__ == '__main__':
    main()
