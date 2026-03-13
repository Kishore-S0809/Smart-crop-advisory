#!/usr/bin/env python3
"""
Utility script to package all CSV dataset files into a single zip archive.

Usage:
    python create_data_zip.py [output_path]

If output_path is omitted the archive is written to data/smart_crop_data.zip.
The resulting zip file can be placed inside the data/ directory and will be
automatically discovered and processed by auto_trainer.py.
"""

import os
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

# File extensions to include in the archive
TABULAR_EXTENSIONS = ('.csv', '.xlsx', '.xls')


def collect_tabular_files(root_dir):
    """Recursively collect all tabular data files under root_dir."""
    collected = []
    for dirpath, _dirs, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(TABULAR_EXTENSIONS):
                collected.append(os.path.join(dirpath, fn))
    return collected


def create_zip(output_path=None):
    """Create a zip archive of all CSV/Excel files in the data directory.

    Falls back to the repository root when the data/ directory does not exist,
    collecting any CSV/Excel files found there.

    Args:
        output_path: Destination path for the zip file. Defaults to
                     ``data/smart_crop_data.zip``.

    Returns:
        The path to the created zip file.
    """
    if output_path is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        output_path = os.path.join(DATA_DIR, 'smart_crop_data.zip')

    # Choose search root
    search_root = DATA_DIR if os.path.isdir(DATA_DIR) else REPO_ROOT

    csv_files = collect_tabular_files(search_root)

    # If data/ was empty, also look in the repo root for loose CSV files
    if search_root == DATA_DIR and not csv_files:
        for fn in os.listdir(REPO_ROOT):
            if fn.lower().endswith(TABULAR_EXTENSIONS):
                csv_files.append(os.path.join(REPO_ROOT, fn))

    if not csv_files:
        print("No tabular data files found to package.")
        return None

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in csv_files:
            # Store files with a path relative to the search root
            arcname = os.path.relpath(file_path, search_root)
            zf.write(file_path, arcname)
            print(f"  Added: {arcname}")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nCreated zip archive: {output_path} ({size_kb:.1f} KB, {len(csv_files)} files)")
    return output_path


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    create_zip(out)
