"""Konfigurasi aplikasi: baca .env, flag, dan tabel level.

Modul ini hanya boleh bergantung pada pustaka standar dan python-dotenv,
supaya aman diimpor oleh modul lain tanpa impor melingkar.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Folder tempat file ini berada (path aset selalu relatif terhadap ini, bukan folder kerja)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# override=True: nilai di .env menang atas variabel yang sudah ada di terminal
load_dotenv(dotenv_path=Path(BASE_DIR) / '.env', override=True)

# ---------------------------------------------------------------- Flag dari .env
# Mode tampilan (default True jika tidak diset)
DISPLAY_HALF = os.getenv('DISPLAY_HALF', 'true').lower() == 'true'
NORMAL_PATTERN = os.getenv('NORMAL_PATTERN', 'true').lower() == 'true'
BUTTON_MODE = os.getenv('BUTTON_MODE', 'false').lower() == 'true'
HIDE_CAMERA = os.getenv('HIDE_CAMERA', 'false').lower() == 'true'
SHOW_ROTATION_COUNT = os.getenv('SHOW_ROTATION_COUNT', 'true').lower() == 'true'
# Kirim file SQL di cron/sql ke PostgreSQL lewat thread latar. false = file tetap ditulis, tidak dikirim.
ENABLE_DB_SYNC = os.getenv('ENABLE_DB_SYNC', 'true').lower() == 'true'

# Jumlah level maksimum (default 8 jika tidak diset atau tidak valid)
try:
    MAX_LEVEL = int(os.getenv('MAX_LEVEL', '8'))
    if MAX_LEVEL < 1 or MAX_LEVEL > 8:
        MAX_LEVEL = 8
        print(f"Warning: MAX_LEVEL must be between 1 and 8. Using default value: {MAX_LEVEL}")
except (ValueError, TypeError):
    MAX_LEVEL = 8
    print(f"Warning: Invalid MAX_LEVEL in .env file. Using default value: {MAX_LEVEL}")

LEVEL_COUNT = 8

try:
    COUNTDOWN_SECONDS = max(1, int(os.getenv('COUNTDOWN_SECONDS', '3')))
except (ValueError, TypeError):
    COUNTDOWN_SECONDS = 3


# ---------------------------------------------------------------- Level
LEVEL_VARIANTS = {
    1: ['1a', '1b', '1c', '1d'],
    2: ['2a', '2b', '2c', '2d'],
    3: ['3a', '3b', '3c', '3d'],
    4: ['4a', '4b', '4c', '4d'],
    5: ['5a', '5b', '5c', '5d'],
    6: ['6a', '6b', '6c', '6d'],
    7: ['7a', '7b', '7c', '7d'],
    8: ['8a', '8b', '8c', '8d']
}


def parse_custom_levels(custom_level_str):
    """Parse and validate custom levels from string format.

    Args:
        custom_level_str: String in format '1a,2b,3c,4d,5d,6c,7b,8a' or empty string

    Returns:
        dict: Dictionary mapping level numbers to their custom variants, or empty dict if invalid
    """
    if not custom_level_str or custom_level_str == '[]':
        return {}

    try:
        # Remove any whitespace and split by comma
        levels = [lvl.strip() for lvl in custom_level_str.split(',') if lvl.strip()]

        # Validate each level
        custom_levels = {}
        for level in levels:
            if len(level) != 2:
                print(f"Invalid level format: {level}. Must be like '1a', '2b', etc.")
                return {}

            level_num = level[0]
            variant = level[1].lower()

            # Validate level number (1-8)
            if not level_num.isdigit() or int(level_num) < 1 or int(level_num) > 8:
                print(f"Invalid level number in {level}. Must be 1-8.")
                return {}

            # Validate variant (a-d)
            if variant not in ['a', 'b', 'c', 'd']:
                print(f"Invalid variant in {level}. Must be a, b, c, or d.")
                return {}

            custom_levels[int(level_num)] = f"{level_num}{variant}"

        return custom_levels

    except Exception as e:
        print(f"Error parsing CUSTOM_LEVEL: {e}")
        return {}


def sanitize_env_value(raw_value):
    """Trim env values and remove inline comments (e.g., '1a,2b # notes')."""
    if raw_value is None:
        return ''
    value = str(raw_value).strip()
    if '#' in value:
        value = value.split('#', 1)[0].strip()
    return value


# Parse and validate custom levels
CUSTOM_LEVEL_STR = sanitize_env_value(os.getenv('CUSTOM_LEVEL', ''))
CUSTOM_LEVELS = parse_custom_levels(CUSTOM_LEVEL_STR)

if CUSTOM_LEVELS:
    print(f"Using custom levels: {CUSTOM_LEVELS}")
else:
    print("No valid custom levels specified, using random variants")

LEVEL_PATHS = {}
for _variants in LEVEL_VARIANTS.values():
    for _variant in _variants:
        _level_num = _variant[0]  # nomor level (1-8)
        if NORMAL_PATTERN:
            LEVEL_PATHS[_variant] = os.path.join(
                BASE_DIR,
                'FILES',
                'TEST_RANDOM_1500x1500',
                f'Lvl {_level_num}{_variant[1]}.png'  # e.g., 'Lvl 1a.png'
            )
        else:
            LEVEL_PATHS[_variant] = os.path.join(
                BASE_DIR,
                'FILES',
                'TEST_RANDOM_OTAK_ATIK',
                f'Lvl {_level_num}{_variant[1]}.PNG'  # e.g., 'Lvl 1a.png'
            )

# Answers for random question
if NORMAL_PATTERN:
    LEVEL_ANSWERS = {
        '1a': [2,1,1,1], '1b': [2,1,1,2], '1c': [1,1,1,2], '1d': [1,2,2,1],
        '2a': [2,1,6,1], '2b': [3,2,1,2], '2c': [2,2,5,6], '2d': [6,2,2,6],
        '3a': [3,5,1,1], '3b': [1,5,1,6], '3c': [2,3,4,1], '3d': [6,5,3,4],
        '4a': [6,2,1,6], '4b': [3,2,2,5], '4c': [1,5,3,1], '4d': [4,2,2,6],
        '5a': [6,3,5,4], '5b': [5,4,6,3], '5c': [3,5,6,4], '5d': [4,4,6,6],
        '6a': [1,4,3,1], '6b': [6,1,5,5], '6c': [5,1,1,4], '6d': [5,4,4,5],
        '7a': [4,5,5,5], '7b': [4,5,5,6], '7c': [2,5,1,5], '7d': [1,2,3,6],
        '8a': [6,5,6,3], '8b': [6,3,4,2], '8c': [5,6,3,5], '8d': [3,6,5,4],
    }
else:
    LEVEL_ANSWERS = {
        '1a': [1,2,1,2], '1b': [2,1,2,1], '1c': [2,1,1,2], '1d': [1,2,2,1],
        '2a': [2,4,1,1], '2b': [1,6,2,2], '2c': [1,2,2,4], '2d': [2,1,1,6],
        '3a': [4,1,5,1], '3b': [6,2,3,2], '3c': [5,1,4,1], '3d': [3,2,6,2],
        '4a': [4,1,1,6], '4b': [6,2,2,4], '4c': [1,5,3,1], '4d': [2,3,5,2],
        '5a': [4,3,5,6], '5b': [6,5,3,4], '5c': [6,6,6,6], '5d': [4,4,4,4],
        '6a': [5,3,4,6], '6b': [3,5,6,4], '6c': [6,6,4,4], '6d': [4,4,6,6],
        '7a': [3,4,6,5], '7b': [5,6,4,3], '7c': [3,6,4,5], '7d': [5,4,6,3],
        '8a': [5,4,4,5], '8b': [3,6,6,3], '8c': [4,5,3,6], '8d': [6,3,5,4],
    }
