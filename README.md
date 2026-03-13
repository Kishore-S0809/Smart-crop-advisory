# Smart-crop-advisory

A machine-learning toolkit for crop yield prediction, fertilizer advisory, and weather forecasting.

## Adding data files

### Using CSV / Excel files directly

Place your CSV or Excel files inside the `data/` directory (any depth). `auto_trainer.py` will discover them automatically.

### Using a zip archive

You can bundle multiple CSV or Excel files into a single `.zip` archive and place it anywhere under `data/`. `auto_trainer.py` automatically scans zip files and reads every CSV/Excel entry inside them.

**Creating the zip archive from existing data files:**

```bash
python create_data_zip.py               # writes data/smart_crop_data.zip
python create_data_zip.py my_data.zip  # custom output path
```

**Using a pre-existing zip:**

Simply copy the zip file into the `data/` directory:

```
data/
└── my_dataset.zip        # auto_trainer.py will read CSVs inside this
```

Then run the trainer as usual:

```bash
python auto_trainer.py
```
