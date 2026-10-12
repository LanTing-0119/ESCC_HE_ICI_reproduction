# ESCC H&E treatment-benefit model — reproduction

Linux notebooks for the manuscript figures and tables. The package demonstrates the complete analysis workflow and supports recalculation with appropriately authorized study data. **Three public TCGA slides and one fabricated clinical row cannot reproduce the trial results.** No trial patient data, paper-trained parameters, private images, or UNI-v2 weights are distributed.

Open the [online report](https://lanting-0119.github.io/ESCC_HE_ICI_reproduction/results/report.html) for the delivered figures, editable slides and validation summary. After cloning or downloading the repository, you can also open `results/report.html` in a local browser. GitHub's file view displays HTML source rather than the rendered report.

The full local/server project retains public synthetic training caches and TCGA image-feature caches. The GitHub repository includes code, the one-row clinical template, public source manifests and rendered demonstration results; WSI files and generated caches are excluded from Git. In a fresh clone, run notebook 02 in demo mode before the clinical notebooks. For spatial figures, download the public slides and run notebook 01 before notebook 08.

## Install and open

```bash
git clone https://github.com/LanTing-0119/ESCC_HE_ICI_reproduction.git
cd ESCC_HE_ICI_reproduction
```


```bash
bash install_linux.sh
conda activate escc_he_ici
jupyter lab notebooks/00_run_selected.ipynb
```

The environment uses Python 3.10 and the numerical versions verified in the existing Linux analysis environment. CUDA 12.8 PyTorch requires a compatible NVIDIA driver; use an allocated GPU for UNI-v2. Statistical notebooks run on CPU. The installation script creates a separate environment and leaves existing environments unchanged. It registers the **ESCC HE ICI** Jupyter kernel.

For the optional synthetic demonstration:

```bash
conda activate escc_he_ici
ESCC_MODE=demo jupyter lab notebooks/00_run_selected.ipynb
```

Notebook 02 generates a sufficient synthetic clinical cohort and independent synthetic patch features, then performs PCA128/K90 training and ten-fold validation. The other clinical notebooks read those generated predictions. All synthetic figures are labeled. Default `template` mode reads the one-row CSV but refuses inferential analysis. Notebook 00 runs only the selected notebooks in the current Python process; it does not start an additional Jupyter kernel or automatically launch WSI tasks.

The clinical simulation is a constructed example, not a statistical replica of the trials. In each cohort, 360 image-included and 30 excluded patients are generated. Treatment alternates 0/1 to give equal arm sizes; it is not assigned by a randomization algorithm. Independent standard-normal variables U and V determine treatment heterogeneity and prognosis. The generating log treatment HR is `-0.35 + 1.1*U`. PFS is exponential with rate `0.065*exp(0.2*V + treatment*(-0.35+1.1*U))`. Latent OS is latent PFS plus an exponential increment with mean `12*exp(-0.2*V)`. A uniform 27–55 month censoring time is shared by PFS/OS. Observed times and event indicators are then formed. These choices intentionally create heterogeneity; they are not estimated from trial patients.

Age is uniform over integers 40–80, sex is 80% male, ECOG is 0/1, and TPS is drawn from 0%, 5%, 30% or missing with probabilities 0.40/0.35/0.20/0.05. With `q=logistic(0.1-treatment*(-0.35+1.1*U))`, response probabilities for CR/PR/SD/PD/NE/missing are `0.05*q, 0.65*q, 0.18, 0.70-0.70*q, 0.07, 0.05`. Demographic fields are independent of the latent prognostic variables.

Ninety random unit-length prototypes in 1536 dimensions generate 24 patch vectors per included synthetic patient. Prototype probabilities are proportional to `exp(linspace(-1,1,90)*U)`; independent Gaussian noise with SD 0.018 is added. Consequently, the constructed patch mixture carries the generating treatment-effect signal. The notebooks then actually train PCA, clusters and Cox models. Displayed HRs and B/NB groups are derived from the fitted models, not copied from the generating formula. The two synthetic cohorts use the same generating mechanism, so this example cannot demonstrate real-world external generalization.

The TCGA spatial demonstration is separate again: its real UNI-v2 features are clustered, but the displayed interaction coefficients are deliberately specified as `0.45*sin(0.51*cluster_index)`, with illustrative SE 0.08. The demonstration reference coefficient and covariance row/column are fixed to zero. No survival model is trained on TCGA outcomes and no TCGA clinical B/NB assignment is made. These coefficients only exercise the visualization code.

For TCGA image processing, obtain authorized UNI-v2 weights from [MahmoodLab](https://huggingface.co/MahmoodLab/UNI2-h), set `UNI_V2_WEIGHTS` to the local checkpoint, optionally set `HEST_WEIGHTS`, and set `ESCC_RUN_WSI=1` before running notebook 01 on a GPU. The default `wsi_patch_limit=0` includes all segmented tissue patches: 2,577, 4,127 and 9,556 for the three examples (16,260 total). The delivered full feature cache was matched by exact coordinates and geometry to an existing public TCGA UNI-v2 cache; all 768 previously extracted example vectors were identical. Details are in `results/tcga_demo/feature_provenance.csv`. Notebook 08 uses these outputs for illustrative spatial maps and an atlas, without inferring clinical outcomes for TCGA cases. A positive `wsi_patch_limit` is an optional quick, sparse demonstration and is stored separately from the full cache.

## Project layout

```text
README.md
environment.yml / requirements.txt / install_linux.sh
config.json
notebooks/       interactive analysis and selected-run entry point
helpers/         input handling, estimators, training, plotting
data/            one-row template, schema and TCGA source manifest (slides downloaded locally)
results/         rendered synthetic/TCGA examples and editable figures; caches stay local
```

## Figure and table map

| Manuscript item | Notebook | Input | Main output |
|---|---|---|---|
| Fig 1 | No analysis notebook | Study design and aggregate counts | results/study_workflow/Figure1 |
| WSI features | 01_wsi_features | Public TCGA slides, authorized UNI-v2 weights | results/tcga_demo/feature_manifest.csv |
| Model and ten-fold predictions | 02_training_and_tenfold | Clinical CSV, patch-feature manifest | results/demo/work or external research directory |
| Tables S1–S3 | 03_clinical_tables | All trial patients, image inclusion, slide mapping | TableS1, TableS2, TableS3 |
| Figs 2–3, S1 | 04_survival_figures | OOF, frozen external, apparent full-data predictions | Figure2, Figure3, FigureS1 |
| Tables 1–2 | 05_absolute_effects | OOF/internal and frozen/external predictions | Table1 (RMST), Table2 (survival probability) |
| Figs S2, S5 | 06_response_and_tps | Response/TPS and appropriate score versions | FigureS2 (ORR/DCR), FigureS5 (TPS) |
| Fig S3 | 07_continuous_interactions | Raw log HR(z), treatment and outcomes | FigureS3 |
| Figs 4, S4 | 08_spatial_and_clusters | TCGA features, coordinates and images | results/tcga_demo/figures/Figure4, FigureS4 |
| Table S4 | 09_fixed_ablations | Same full-data and fold-specific representations | TableS4 |
| Selected tasks | 00_run_selected | User-selected notebooks | Corresponding outputs |

Clinical figure/table outputs are under `results/demo/figures_tables/` in demo mode. Statistics are saved as CSV **before** plotting. Figures/tables provide PNG, PDF and SVG. The delivered figures also have editable PPTX copies; conversion utilities are intentionally not part of the public analytical notebooks. Photographic tissue pixels remain raster; labels, axes, curves, diagrams and map overlays are editable vector objects.

## Fixed analytical choices

- UNI-v2, 20×, 256 pixels, 50% overlap; L2 patch normalization, patient-balanced lower-median P50 sampling, randomized PCA128, K90 MiniBatchKMeans. No color normalization. Slide counts use a 0.5 pseudocount; patient proportions are equal-weight means of slides. CLR is centered using training patients only.
- Full PFS Cox model: prognostic α, treatment βT, interaction γ; ridge 0.01 and robust variance. The reference cluster has the largest mean training proportion (C15 for the original full-data model). B: HR(z)<1; NB: HR(z)≥1. The threshold is fixed.
- Stratified ten-fold splits use treatment × PFS event and a fixed computational seed. Every fold refits representation and model using its training patients only. The external cohort never supplies fitting information. Candidate selection is not repeated within folds; this evaluates the fixed candidate, not the entire selection procedure.
- Group-specific HR: ICI+CT / CT. Binary interaction: Altman–Bland HR B / HR NB. Only internal OOF Cox evaluations stratify by fold. Apparent and external evaluations do not.
- Continuous evaluation: unpenalized robust Cox with treatment, **raw log HR(z)** and their interaction. Interaction HR is per one raw score unit. OS evaluates a PFS-trained score.
- Absolute differences: ICI+CT minus CT; between-group difference: B treatment difference minus NB treatment difference. PFS horizon 12 months, OS horizon 24 months. RMST is in months; survival differences are percentage points. Greenwood plug-in Wald inference conditions on fixed groups.
- ORR=CR+PR; DCR=CR+PR+SD. Missing/NE remain in primary denominators. Evaluability sensitivity is saved separately. Error bars are one binomial SE. TPS tests are nominal Pearson tests without continuity or multiplicity correction. TPS uses full-data groups in JUPITER, as in the manuscript.
- P values are nominal and two-sided. Selection and overlapping CV-training uncertainty are not included in the reported conditional intervals. External stratification evidence does not establish that NB patients can safely omit ICI.

## Authorized study-data recalculation

Keep the clinical CSV, feature/slide manifest, research configuration and **all research outputs outside this project**. Set `ESCC_CONFIG` to an external JSON file overriding `mode` (`research`), `clinical_csv`, `slide_manifest`, `output_dir`, and `enable_large_tasks` (`true` only when a full training run is intended). Use absolute paths in that private file. No private configuration is copied into the public package. See [data/README.md](data/README.md) for fields and data access.

Start with notebook 02 for model reconstruction, then run the required analytical notebooks. Original preprocessing, clinical versions, patient ordering, slide matching and software versions must be preserved for exact numerical reconstruction. Existing restricted fold caches can be used privately for independent verification; they are not bundled. Public training outputs generated from synthetic data are distinguishable by `SIM_` identifiers.

For research Figure 4/S4, additionally provide an external `pathology_manifest` with selected patient_id, slide_id, wsi_path, feature_path, coords_path, panel (A/B) and pfs_category (short/long) columns. Retain the original authorized eight case choices; panel A contains B/short and NB/long cases and panel B contains B/long and NB/short cases. The code checks ICI+CT treatment and the within-cohort observed-PFS quartile rule, including an event for short-PFS cases. Notebook 08 reads the reconstructed full-data model and its robust interaction covariance. Patient HRs come from all-slide patient predictions, not the displayed slide alone. `atlas_clusters` can specify the clusters to illustrate; representative patches are selected from the supplied image manifest. Exact paper case/atlas reconstruction requires the original authorized case choices and representative-image pool.

For spatial maps, the regional coefficient is the mean of the original reference-coded interaction coefficients of covering patches. Its variance is `w.T @ covariance @ w`, with w equal to cluster frequencies among those patches. There is no across-cluster mean subtraction. Directions are relative to the model's reference cluster (C15 in the paper). Nominal two-sided Wald P<0.05 determines negative/positive associations; gray is nonsignificant/nonestimable, and white is uncovered. Area percentages use the patch union and count overlaps once. These categories are not local B/NB labels.


## Method references

Altman DG, Bland JM. Interaction revisited: the difference between two estimates. BMJ. 2003;326:219. [doi:10.1136/bmj.326.7382.219](https://doi.org/10.1136/bmj.326.7382.219).

Lin DY, Wei LJ. The robust inference for the Cox proportional hazards model. JASA. 1989;84:1074–1078. [doi:10.1080/01621459.1989.10478874](https://doi.org/10.1080/01621459.1989.10478874).

The repository organization is inspired by [S2Omics](https://github.com/ddb-qiwang/S2Omics); no S2Omics model or analysis code is used. WSI processing uses [TRIDENT](https://github.com/mahmoodlab/TRIDENT). Respect the licenses and access conditions of third-party code, models and TCGA data. No additional reuse license for the study authors' analysis code is granted by this repository. Contact the authors for reuse permission; third-party licenses remain applicable.
