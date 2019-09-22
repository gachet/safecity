import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(ROOT_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
