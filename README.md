# Smart-crop-advisory

A machine learning–based system for smart crop advisory, including crop recommendation, fertilizer advisory, and weather forecasting models.

## Repository Contents

| File / Script | Description |
|---|---|
| `auto_trainer.py` | Trains ML models on the included CSV datasets |
| `bundle_models.py` | Selects the best-performing models and bundles them |
| `api_demo.py` | Demonstrates how to query the bundled models |
| `create_zip.py` | Creates a zip archive of datasets and/or scripts |
| `*.csv` | Raw datasets used for training |

## Creating a Zip Archive

Use `create_zip.py` to package the datasets (and optionally the scripts) into a distributable zip file.

### Archive datasets only (default)

```bash
python create_zip.py
```

This creates a timestamped file such as `smart_crop_advisory_datasets_20240101_120000.zip` containing all CSV files.

### Archive datasets and scripts

```bash
python create_zip.py --mode full
```

### Specify a custom output path

```bash
python create_zip.py --output my_archive.zip
python create_zip.py --mode full --output my_archive.zip
```

### List the contents of an existing zip

```bash
python create_zip.py --list my_archive.zip
```

## Training Models

```bash
python auto_trainer.py
```

## Bundling Best Models

```bash
python bundle_models.py
```

## API Demo

```bash
python api_demo.py
```