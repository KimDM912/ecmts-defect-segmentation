# Expected Confusion Matrix-Based Deployment-Time Threshold Selection for Pre-trained Defect Segmentation Models in Manufacturing Inspection

This repository contains the implementation of **Expected Confusion Matrix-based Threshold Selection (ECMTS)** for binary defect segmentation.

ECMTS selects an image-wise binarization threshold from a model-produced probability map. It constructs plug-in expected confusion counts and chooses the threshold that maximizes the expected image-wise \(F_2\)-score. Ground-truth masks are not used to select thresholds for test images.

## Supported experiments

### Datasets

- Severstal Steel Defect Detection (Severstal)
- Kolektor Surface-Defect Dataset 2 (KolektorSDD2)
- Magnetic Tile Defect Dataset (MTD)

### Segmentation backbones

- ResNet-18
- ResNet-34
- EfficientNet-B0

Each backbone is used as an ImageNet-pretrained encoder in a shared U-Net segmentation architecture.

### Threshold selection methods

- Fixed threshold of 0.5
- Validation-based threshold selection \(F_2\)
- Otsu thresholding
- Kapur thresholding
- ECMTS

## Repository structure

```text
ecmts-defect-segmentation/
├── analysis/                   # Oracle-regret, agreement, and runtime analyses
├── data/
│   ├── evaluation_manifests/   # Defect-only validation/test cohort manifests
│   └── splits/                 # Reproducible train/validation/test splits
├── data_cohorts/               # Defect-cohort generation scripts
├── data_split/                 # Dataset split generation scripts
├── evaluation/
│   ├── main_experiment/        # Threshold selection and evaluation
│   └── summarize_thresholding_results.py
└── training/                   # U-Net training scripts
```

Run all commands from the repository root unless stated otherwise.

## Environment setup

A CUDA-capable GPU is recommended for model training and evaluation. Install a PyTorch build compatible with the local CUDA or CPU environment, and then install the remaining dependencies.

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch torchvision
pip install numpy pillow pandas scipy matplotlib openpyxl tensorboard
```

## Dataset preparation

The original datasets are not redistributed in this repository. Obtain them from their original public sources and place them under `data/` using the following default layout:

```text
data/
├── severstal-steel-defect-detection/
│   ├── train_images/
│   └── train.csv
├── kolektor/
│   ├── train/
│   └── test/
└── magnetic-tile-defect-datasets.-master/
    ├── MT_Blowhole/
    ├── MT_Break/
    ├── MT_Crack/
    ├── MT_Fray/
    ├── MT_Free/
    └── MT_Uneven/
```

The repository already includes the split JSON files used in the experiments. Because image and mask paths are stored relative to each dataset root, the supplied manifests can be used after the datasets are placed in the expected directories.

Custom dataset locations can also be supplied through each script's `--data-root` argument.

## 1. Generate dataset splits

This step is optional when using the split manifests already included in `data/splits/`.

```bash
python data_split/generate_split_severstal.py
python data_split/generate_split_kolektor.py
python data_split/generate_split_magnetic.py
```

Default split policies:

| Dataset | Split policy |
|---|---|
| Severstal | 64% training, 16% validation, and 20% test; stratified by defect-class combination |
| KolektorSDD2 | Official test set retained; official training set divided into 80% training and 20% validation |
| Magnetic Tile | 64% training, 16% validation, and 20% test; stratified by dataset domain |

For Severstal, the split-generation script also creates binary ground-truth masks by combining the positive RLE annotations from defect classes 1–4.

## 2. Generate evaluation cohorts

The thresholding experiment uses defect-containing validation and test images. The following scripts inspect the processed ground-truth masks and create the corresponding cohort manifests without copying or modifying images:

```bash
python data_cohorts/generate_defect_cohorts_severstal.py
python data_cohorts/generate_defect_cohorts_kolektor.py
python data_cohorts/generate_defect_cohorts_magnetic.py
```

The generated files are stored in:

```text
data/evaluation_manifests/
```

The supplied cohort manifests can be used directly when the dataset layout matches the default paths.

## 3. Train segmentation models

A single dataset–backbone combination can be trained with:

```bash
python training/train_segmentation.py --dataset severstal --backbone resnet18
```

Available dataset identifiers:

```text
severstal
kolektor
magnetic
```

Available backbone identifiers:

```text
resnet18
resnet34
efficientnet_b0
```

Examples:

```bash
python training/train_segmentation.py --dataset severstal --backbone resnet34
python training/train_segmentation.py --dataset kolektor --backbone efficientnet_b0
python training/train_segmentation.py --dataset magnetic --backbone resnet18
```

Dataset- and backbone-specific launcher scripts are also provided in `training/`.

The default training protocol uses:

- random seed 2026;
- binary cross-entropy over valid image regions;
- AdamW with a learning rate of `3e-4` and weight decay of `1e-4`;
- a maximum of 60 epochs;
- validation-BCE-based learning-rate reduction and early stopping;
- the checkpoint with the lowest validation BCE.

Default input processing:

| Dataset | Input size | Processing |
|---|---:|---|
| Severstal | 256 × 1600 | Native image canvas |
| KolektorSDD2 | 640 × 384 | Aspect-ratio-preserving letterbox resize |
| Magnetic Tile | 512 × 512 | Aspect-ratio-preserving letterbox resize |

Padded regions are excluded from the training loss and evaluation metrics.

The best checkpoint is saved to:

```text
checkpoints/<dataset>/<backbone>/seed_2026/best.pth
```

## 4. Run the thresholding experiment

Evaluate one dataset–backbone combination with:

```bash
python evaluation/main_experiment/evaluate_thresholding.py \
    --dataset severstal \
    --backbone resnet18
```

Dataset- and backbone-specific launcher scripts are also available in `evaluation/main_experiment/`, for example:

```bash
python evaluation/main_experiment/run_severstal_resnet18.py
python evaluation/main_experiment/run_kolektor_resnet34.py
python evaluation/main_experiment/run_magnetic_efficientnet_b0.py
```

The default evaluation uses:

- an ECMTS threshold grid of 2,001 values from 0 to 1;
- 256 histogram bins for Otsu and Kapur thresholding;
- image-wise macro Precision, Recall, and \(F_2\);
- validation ground truth only for selecting the global Validation-\(F_2\) threshold;
- no test ground truth for threshold selection;
- image-wise oracle results only for diagnostic analysis.

Results are written to:

```text
evaluation/results_thresholding/<dataset>/<backbone>/seed_2026/
```

Each run produces:

```text
validation_f2_curve.csv
validation_threshold_selection.json
per_image_metrics.csv
selected_thresholds.csv
diagnostics.csv
aggregate_metrics.csv
run_metadata.json
```

A small smoke test can be run with deterministic subsets:

```bash
python evaluation/main_experiment/evaluate_thresholding.py \
    --dataset severstal \
    --backbone resnet18 \
    --max-validation-images 5 \
    --max-test-images 5
```

## 5. Summarize the main results

After running all requested dataset–backbone combinations, create the aggregate Excel workbook with:

```bash
python evaluation/summarize_thresholding_results.py
```

The default output is:

```text
evaluation/results_thresholding/thresholding_method_summary.xlsx
```

The workbook contains:

- image-wise macro Precision, Recall, and \(F_2\) summaries;
- paired image-level sign-flip permutation tests;
- paired bootstrap confidence intervals;
- Shapiro–Wilk tests of paired \(F_2\) differences.

The permutation and bootstrap procedures use 10,000 iterations by default. No multiple-comparison correction is applied by this script.

## 6. Run additional analyses

### Global-threshold oracle regret

```bash
python analysis/analyze_global_oracle_regret.py
```

This analysis compares the validation-derived global threshold with the image-wise oracle threshold using paired image-level \(F_2\) regret.

### Expected–true \(F_2\) agreement

```bash
python analysis/expected_true_correlation/analyze_expected_true_correlation.py
```

This script summarizes the agreement between expected and true \(F_2\), including Pearson correlation, Spearman correlation, threshold gaps, and oracle regret.

### Runtime benchmark

```bash
python analysis/runtime/benchmark_threshold_runtime.py
```

By default, the runtime script evaluates 50 deterministically selected test images per dataset–backbone combination and reports inference and threshold-selection time.

## Reproducibility notes

- The default random seed is `2026`.
- Split and cohort manifests used in the experiments are included in the repository.
- Checkpoints, original datasets, and generated evaluation outputs are not included.
- Run metadata and image-level results are saved for each thresholding experiment.
- Paths in the supplied split manifests are relative to the corresponding dataset root.

## Citation

Citation information will be added after publication.

## License

A software license has not yet been included. Add a license file before distributing or permitting reuse of the code.
