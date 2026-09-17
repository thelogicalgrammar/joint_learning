# Data

`langlearning_v2_anonymized.csv` is the behavioural dataset of the Aclapa
word-learning experiment (386 Prolific participants, 200 trials each, run
11–23 August 2021 on PCIbex), in PCIbex's long "results" format: one row
per recorded field, columns `Results index` (participant), `Time`,
`Counter`, `Hash`, ..., `Type`, `Field name`, `Field value`. All loaders in
this repository (`jointlearn/data.py`, the R script in
`analyses/model_free/`, `jointlearn/hmm/dataset.py`) read this
file by default.

It is derived from the raw PCIbex export by `anonymize.py`, which removes
the personal data: Prolific participant IDs, the free-text `comments` and
`strategy` answers, the PCIbex session hash, the time of day (dates are
kept), and any value that looks like a Prolific ID typed into another box.
The demographic answers used by the model-free analysis (age, gender,
native / daily / second / later language, and a yes/no `notes` field) are
kept. The experimental records (instructions, trial-by-trial choices,
reaction times, word-order box, lexicon test) are unchanged, so every
analysis array is identical to the one obtained from the raw file.

The raw export contains identifiable data and is **not** part of the
repository; it is kept outside the repository directory. To regenerate the
anonymized file: `python data/anonymize.py /path/to/raw.csv`.
