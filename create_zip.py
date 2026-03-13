#!/usr/bin/env python3
"""
Utility script to create a zip archive of the Smart Crop Advisory datasets and scripts.
"""

import os
import zipfile
from datetime import datetime

# Paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# File patterns to include in the zip archive
CSV_EXTENSIONS = {'.csv'}
SCRIPT_EXTENSIONS = {'.py', '.txt'}

def create_datasets_zip(output_path: str = None) -> str:
    """
    Create a zip archive containing all CSV dataset files.

    Args:
        output_path: Path for the output zip file. Defaults to
                     'smart_crop_advisory_datasets_<timestamp>.zip' in the repo root.

    Returns:
        Path to the created zip file.
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(REPO_ROOT, f'smart_crop_advisory_datasets_{timestamp}.zip')

    csv_files = [
        f for f in os.listdir(REPO_ROOT)
        if os.path.isfile(os.path.join(REPO_ROOT, f))
        and os.path.splitext(f)[1].lower() in CSV_EXTENSIONS
    ]

    if not csv_files:
        print("No CSV dataset files found.")
        return output_path

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(csv_files):
            file_path = os.path.join(REPO_ROOT, filename)
            zf.write(file_path, filename)
            print(f"  Added: {filename}")

    print(f"\nDataset archive created: {output_path}")
    print(f"Total files archived: {len(csv_files)}")
    return output_path


def create_full_zip(output_path: str = None) -> str:
    """
    Create a zip archive containing all CSV datasets and Python scripts.

    Args:
        output_path: Path for the output zip file. Defaults to
                     'smart_crop_advisory_<timestamp>.zip' in the repo root.

    Returns:
        Path to the created zip file.
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(REPO_ROOT, f'smart_crop_advisory_{timestamp}.zip')

    included_extensions = CSV_EXTENSIONS | SCRIPT_EXTENSIONS
    files_to_archive = [
        f for f in os.listdir(REPO_ROOT)
        if os.path.isfile(os.path.join(REPO_ROOT, f))
        and os.path.splitext(f)[1].lower() in included_extensions
    ]

    if not files_to_archive:
        print("No files found to archive.")
        return output_path

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(files_to_archive):
            file_path = os.path.join(REPO_ROOT, filename)
            zf.write(file_path, filename)
            print(f"  Added: {filename}")

    print(f"\nFull archive created: {output_path}")
    print(f"Total files archived: {len(files_to_archive)}")
    return output_path


def list_zip_contents(zip_path: str) -> None:
    """
    List the contents of an existing zip archive.

    Args:
        zip_path: Path to the zip file.
    """
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        return

    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        print(f"Contents of {os.path.basename(zip_path)} ({len(infos)} files):")
        for info in infos:
            size_kb = info.file_size / 1024
            print(f"  {info.filename:50s}  {size_kb:8.1f} KB")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a zip archive of Smart Crop Advisory datasets and/or scripts.'
    )
    parser.add_argument(
        '--mode',
        choices=['datasets', 'full'],
        default='datasets',
        help='datasets: archive only CSV files (default); full: archive CSV files and Python scripts.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output zip file path (optional). Defaults to a timestamped filename in the repo root.'
    )
    parser.add_argument(
        '--list',
        metavar='ZIP_PATH',
        help='List the contents of an existing zip file and exit.'
    )
    args = parser.parse_args()

    if args.list:
        list_zip_contents(args.list)
    elif args.mode == 'full':
        print("Creating full archive (datasets + scripts)...")
        create_full_zip(args.output)
    else:
        print("Creating datasets archive...")
        create_datasets_zip(args.output)
