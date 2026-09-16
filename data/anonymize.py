"""
Build the anonymized experiment dataset that the repository ships
(data/langlearning_v2_anonymized.csv) from the raw PCIbex results file,
which contains personal data and must stay outside the repository.

Removed from the raw file:
  - the `prolific` rows (Prolific participant IDs: direct identifiers)
  - the free-text rows `comments` and `strategy`
  - any remaining value that looks like a Prolific ID (24 hex characters;
    a few participants typed their ID into a demographic box)
  - the PCIbex session hash (column `Hash`, blanked)
  - the time of day (column `Time` is reduced to the date)
Kept: every experimental row (instructions, game trials, word-order box,
lexicon test, reaction times) and the demographic answers the model-free
analysis uses (age, gender, native / daily / second / later language,
`notes` = yes/no). All loaders index columns by position, so the column
layout is unchanged.

Usage:  python data/anonymize.py [path/to/raw.csv]
Default raw path: ../2022_joint_learning_private_data/michael_data/data/
results_2021-08-23T12_58_44_175Z_langlearning-v2.csv (next to the repo).
"""
import sys
import hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DEFAULT = (HERE.parent.parent / '2022_joint_learning_private_data' / 'michael_data' / 'data'
               / 'results_2021-08-23T12_58_44_175Z_langlearning-v2.csv')
OUT = HERE / 'langlearning_v2_anonymized.csv'
DROP_FIELDS = {'prolific', 'comments', 'strategy'}

raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DEFAULT
df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
n0 = len(df)
df = df[~df['Field name'].isin(DROP_FIELDS)].copy()
df['Hash'] = ''
id_like = df['Field value'].str.fullmatch(r'\s*[0-9a-f]{24}\s*')
df.loc[id_like, 'Field value'] = '[redacted]'
df['Time'] = df['Time'].str.slice(0, 10)
df.to_csv(OUT, index=False)
print(f'{raw_path.name}: {n0} rows -> {OUT.name}: {len(df)} rows '
      f'({n0 - len(df)} dropped: ' + ', '.join(sorted(DROP_FIELDS)) + f'; '
      f'{int(id_like.sum())} ID-like values redacted); '
      f'sha1 {hashlib.sha1(OUT.read_bytes()).hexdigest()[:12]}')
