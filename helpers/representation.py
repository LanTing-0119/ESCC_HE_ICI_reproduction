"""Training-only patch sampling, L2 normalization, PCA and clustering."""
from pathlib import Path
import json, math, warnings, hashlib, tempfile
import numpy as np
import pandas as pd
import h5py, joblib
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import os
from contextlib import nullcontext

_DEMO_FEATURE_CACHE = {}

def feature_file(path):
    """Cache only the small synthetic example; real WSI features remain disk-backed."""
    if os.environ.get('ESCC_MODE') != 'demo':
        return h5py.File(path, 'r')
    key = str(path)
    if key not in _DEMO_FEATURE_CACHE:
        with h5py.File(path, 'r') as h:
            _DEMO_FEATURE_CACHE[key] = h['features'][:]
    return nullcontext({'features': _DEMO_FEATURE_CACHE[key]})
def seed(base, *parts):
    """Derive a deterministic 64-bit seed from a base seed and identifiers using SHA-256."""
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (base, *parts))).encode()).digest()[:8], 'little')

def quotas(available, cap):
    """For patients with multiple slides, quotas allocates the patient-level sampling budget across those slides."""
    target = min(cap, sum(available.values()))
    result = dict.fromkeys(available, 0)
    active = sorted((k for (k, n) in available.items() if n > 0))
    left = target
    while left and active:
        share = max(1, left // len(active))
        for key in active:
            take = min(available[key] - result[key], share, left)
            result[key] += take
            left -= take
        active = [k for k in active if result[k] < available[k]]
    assert sum(result.values()) == target
    return result

def patch_plan(manifest, patients, cap, random_seed):
    """Turns a per-patient sampling cap into a concrete, reproducible list of which patches to sample from each slide."""
    rows = []
    for (patient, group) in manifest[manifest.patient_id.isin(patients)].groupby('patient_id', sort=True):
        allocation = quotas(dict(zip(group.slide_id, group.n_patches)), cap)
        for row in group.sort_values('slide_id').itertuples():
            n = allocation[row.slide_id]
            rng = np.random.default_rng(seed(random_seed, patient, row.slide_id))
            indices = np.arange(row.n_patches) if n == row.n_patches else np.sort(rng.choice(row.n_patches, n, replace=False))
            rows.append(pd.DataFrame({'patient_id': patient, 'slide_id': row.slide_id, 'feature_path': row.feature_path, 'patch_index': indices}))
    plan = pd.concat(rows, ignore_index=True)
    assert not plan.duplicated(['slide_id', 'patch_index']).any()
    return plan

def matrix_from_plan(plan, destination, dim):
    """Load preselected patch features into an L2-normalized, disk-backed matrix."""
    matrix = np.lib.format.open_memmap(destination, mode='w+', dtype='float32', shape=(len(plan), dim))
    offset = 0
    for (path, group) in plan.groupby('feature_path', sort=False):
        idx = group.patch_index.to_numpy(dtype=np.int64)
        assert np.all(np.diff(idx) > 0)
        with feature_file(path) as f:
            features = f['features']
            for block in np.unique(idx // 4096):
                chosen = idx[idx // 4096 == block]
                start = int(block * 4096)
                x = np.asarray(features[start:min(start + 4096, len(features))], dtype='float32')[chosen - start]
                if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
                    raise ValueError('Nonfinite or zero feature vector; no silent exclusion permitted')
                
                # L2 Normalization
                matrix[offset:offset + len(x)] = normalize(x, copy=False)
                offset += len(x)
        pass
    assert offset == len(plan)
    matrix.flush()
    return matrix

def balanced_batches(total, preferred=8192, minimum=128):
    """Return balanced batch sizes that sum to total and are at least minimum."""
    if total < minimum:
        raise ValueError('Insufficient samples')
    n = max(1, math.ceil(total / preferred))
    (base, rem) = divmod(total, n)
    sizes = [base + (i < rem) for i in range(n)]
    if min(sizes) < minimum:
        n = max(1, total // minimum)
        (base, rem) = divmod(total, n)
        sizes = [base + (i < rem) for i in range(n)]
    assert sum(sizes) == total and min(sizes) >= minimum
    return sizes

def map_all(manifest, pca, kmeans, outdir):
    """Assign all patches from each slide to clusters using trained PCA and KMeans, and then aggregate them into cluster proportion features for each patient."""
    records = []
    k = kmeans.n_clusters

    # Read all patch features slide by slide
    for row in manifest.sort_values(['patient_id', 'slide_id']).itertuples():
        path = Path(row.feature_path)
        with feature_file(path) as h:
            assert h['features'].shape == (row.n_patches, len(pca.mean_))
            x = np.asarray(h['features'][:], dtype=np.float32)
        assert np.isfinite(x).all() and (not np.any(np.linalg.norm(x, axis=1) == 0))

        # Normalize, reduce dimensionality, and compute the distances to each cluster center.
        with threadpool_limits(limits=1):
            normalized = normalize(x, norm='l2', axis=1, copy=False) # normalize: M × 1536
            z = pca.transform(normalized).astype('float32') # reduce dimensionality: M × 128
            distances = kmeans.transform(z) # compute the distances to each cluster center: M × 90

        # define smallest distance as the cluster ID of that patch
        labels = distances.argmin(axis=1)
        counts = np.bincount(labels, minlength=k)
        assert int(counts.sum()) == row.n_patches
        records.append(dict(patient_id=row.patient_id, slide_id=row.slide_id, total_patches=row.n_patches, mapped_patches=int(counts.sum()), **{f'count_cluster_{i:02d}': int(v) for (i, v) in enumerate(counts)}))
        del x, normalized, z, distances, labels
        if len(records) % 50 == 0:
            print('Slides processed', len(records), '/', len(manifest), flush=True)
    
    counts = pd.DataFrame(records)
    counts.to_csv(outdir / 'slide_cluster_counts.csv', index=False)

    # Convert the patches counts of each cluster into smoothed proportions.
    cc = [f'count_cluster_{i:02d}' for i in range(k)]
    pc = [f'prop_cluster_{i:02d}' for i in range(k)]
    smoothed = (counts[cc].to_numpy() + 0.5) / (counts.total_patches.to_numpy()[:, None] + 0.5 * k)
    slides = pd.DataFrame(smoothed, columns=pc)
    slides.insert(0, 'slide_id', counts.slide_id)
    slides.insert(0, 'patient_id', counts.patient_id)
    slides.to_csv(outdir / 'slide_cluster_proportions.csv', index=False)
    patients = slides.groupby('patient_id', sort=True)[pc].mean()
    assert np.allclose(patients.sum(axis=1), 1) and (patients.to_numpy() > 0).all()
    return (patients, counts)

def reference_and_center(proportions, training_ids):
    """Convert the cluster proportions into CLR features, centering with the mean of the training patients, and determine the reference cluster."""
    raw = np.log(proportions)
    raw = raw.sub(raw.mean(axis=1), axis=0)
    center = raw.loc[training_ids].mean(axis=0).to_numpy()
    reference = int(np.argmax(proportions.loc[training_ids].mean(axis=0).to_numpy()))

    # z: Matrix of patient-level CLR features mean-centered using the training set means
    z = pd.DataFrame(raw.to_numpy() - center, index=raw.index, columns=[f'clr_cluster_{i:02d}' for i in range(raw.shape[1])])
    return (z, center, reference)

def fit_representation(manifest, clinical, output, random_seed=20260830, pca_dim=128, n_clusters=90):
    """Learn a shared dimensionality reduction rule and morphological clustering system from the existing UNI-v2 patch features of the training patients."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(clinical.patient_id)

    # Set the per-patient sampling cap to the lower 50th percentile of total patch counts per patient in the manifest.
    manifest = manifest[manifest.patient_id.isin(ids)].copy()
    cap = int(np.quantile(manifest.groupby('patient_id').n_patches.sum(), 0.5, method='lower'))

    # Select patch indices for clinical patients, subject to the per-patient cap.
    plan = patch_plan(manifest, ids, cap, random_seed)
    saved_plan=plan.copy()
    if os.environ.get('ESCC_MODE')=='demo':
        project=Path(__file__).resolve().parents[1]
        saved_plan['feature_path']=saved_plan.feature_path.map(lambda p:str(Path(p).relative_to(project)))
    saved_plan.to_csv(output / 'sampled_patch_indices.csv.gz', index=False)

    # Group selected patches by feature file and sort indices for block-wise reading.
    stream = plan.sort_values(['feature_path', 'patch_index']).reset_index(drop=True) 

    with tempfile.TemporaryDirectory(prefix='representation_', dir=output) as folder, threadpool_limits(limits=1): # underlying libraries enable multithreading by default
        folder = Path(folder)

        # Load preselected patch features, apply per-patch L2 normalization and store them in a disk-backed matrix.
        x = matrix_from_plan(stream, folder / 'normalized.npy', 1536)
        print(f'Fitting PCA on {len(plan):,} sampled patches (P50 cap={cap}).')

        # Fit PCA using randomized SVD to reduce features to 128 dimensions
        pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=random_seed).fit(x)
        
        # Apply the fitted PCA transformation in batches
        reduced = np.lib.format.open_memmap(folder / 'reduced.npy', mode='w+', dtype='float32', shape=(len(x), pca_dim))
        start = 0
        for size in balanced_batches(len(x)):
            reduced[start:start + size] = pca.transform(x[start:start + size]).astype('float32')
            start += size
        reduced.flush()

        # Train KMeans on the 128-dimensional features.
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=8192, n_init=10, max_iter=300, random_state=random_seed).fit(reduced)
        
        # Save the fitted PCA and KMeans models for reuse.
        joblib.dump(pca, output / 'pca.joblib')
        joblib.dump(kmeans, output / 'kmeans.joblib')
        del x, reduced
    return (pca, kmeans)
