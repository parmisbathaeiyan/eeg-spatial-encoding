# TeCo data — structure reference (Task 1)

A complete description of the TeCo `.pickle` files in this project, written so that
someone (or an agent) with zero prior context can load and use them correctly.
Everything below was verified against the actual files.

---

## 1. What TeCo is

**TeCo** is a Persian (Farsi) dataset of **simultaneous EEG + eye-tracking** recorded
while 12 people read sentences. It is the Persian counterpart of the English **ZuCo**
dataset (see §8) and was built for the same purpose: using brain/gaze signals to study
and improve NLP models. It was collected for Yaser Abbaszadeh's MSc thesis (University
of Tehran, 2023).

There are three reading tasks. **This document covers Task 1 only** (the others are out
of scope here):

| Task | Name | Content | # sentences |
|------|------|---------|-------------|
| **Task 1** | SR — Sentiment Reading | Persian translations of English movie-review sentences (from SST) | **165** |
| Task 2 | NR — Normal Reading | Wikipedia sentences | 160 |
| Task 3 | TSR — Task-Specific Reading | Wikipedia sentences, relation extraction | 285 |

**Recording**: 128-channel g.HIAMP EEG cap (g.GAMMASYS), 1200 Hz, at the National Brain
Mapping Lab; eye movement from a monocular eye-tracker at 500 Hz. After dropping 2
reference electrodes, **126 EEG channels** remain — this is why `126` appears everywhere.

**Sentiment labels** (Task 1): each of the 165 sentences is positive / negative / neutral,
perfectly balanced **55 / 55 / 55**. 15 of the 165 are *control* sentences (had a
comprehension question); they are split 5/5/5 across sentiments.

---

## 2. The files you have

Data root: `/Users/parmis/Documents/thesis_data/` (on Colab, wherever you upload it).

```
thesis_data/
├── TRT_Total/                         # SMALL pickles (one field extracted)
│   ├── Amin_trt_total.pickle ...      # 12 subjects, ~1–1.5 MB each
├── TeCo/
│   ├── task1_with_total/              # FULL pickles (everything)
│   │   ├── armin/task1.{pickle,mat}   # 12 subjects, ~1.3–2.1 GB each
│   │   ├── arya/  ... etc
│   └── sentences/Task_1/*.xlsx        # original stimulus/label spreadsheets
└── teco_sentiment_labels_task1.csv    # clean label file (recommended)
```

**Channel locations** (for topographic plots / spatial analysis): the 126 electrodes'
3-D coordinates live in a separate `location_xyz_reference.txt` (`index, x, y, z, label`,
tab-separated). It is **not in `thesis_data`** — it's kept on Drive. Load it into an MNE
montage to draw topomaps:

```python
loc = pd.read_csv("location_xyz_reference.txt", sep="\t", header=None,
                  names=["index", "x", "y", "z", "label"])
montage = mne.channels.make_dig_montage(
    ch_pos={r.label: [r.x, r.y, r.z] for r in loc.itertuples()}, coord_frame="head")
```

There are **two pickle variants**, both keyed the same way (see §3):

- **`TRT_Total/<Name>_trt_total.pickle`** — a *thin extract*: only the per-word
  `TRT_total` field. ~1 MB. Has all **12** subjects. Use this when you just need the
  full-band total-reading-time EEG per word (e.g. the EEG-only / multimodal sentiment
  models). Verified: its `word['TRT_total']` is **byte-identical** to the same field in
  the full pickle.
- **`task1_with_total/<name>/task1.pickle`** — the *full superset* the thin files were
  extracted from: raw EEG, all eye-tracking measures, all frequency bands, fixations,
  control-answer EEG, both English and Persian text. ~2 GB. Now has all **12** subjects.

> `.mat` versions exist alongside the full `.pickle` (same content, MATLAB format). The
> `.pickle` is easier from Python; prefer it.

**Subject name mapping** — the two pickle sets spell names differently, so be careful:

| Code | Age/Sex | `TRT_Total` filename | full-pickle folder |
|------|---------|----------------------|--------------------|
| AMB | 29 M | `Amin_trt_total.pickle`     | `armin/` |
| ARM | 25 M | `Arya_trt_total.pickle`     | `arya/` |
| EIA | 29 M | `Eisa_trt_total.pickle`     | `eisa/` |
| ELD | 30 F | `Elahe_trt_total.pickle`    | `elahe/` |
| FAS | 24 M | `Farzam_trt_total.pickle`   | `farzam/` |
| JAY | 26 M | `Javad_trt_total.pickle`    | `javad/` |
| MEB | 27 F | `Mehrnaz_trt_total.pickle`  | `mehrnaz/` |
| NAD | 26 F | `Nastaran_trt_total.pickle` | `nastaran/` |
| RAS | 29 F | `Razieh_trt_total.pickle`   | `razie/` |
| SAS | 26 M | `Sajad_trt_total.pickle`    | `sajad/` |
| SHU | 29 F | `Shahrzad_trt_total.pickle` | `sharzad/` |
| AYB | 29 M | `Yaser_trt_total.pickle`    | `yaser/` |

---

## 3. Top-level structure (both pickle variants)

A pickle loads to a **dict of 165 trials**, keyed by integer **0..164** (trial order).
Trial order follows `sentenceId` 1..165, so `trial_key = sentenceId - 1`.

```python
import pickle
with open("Amin_trt_total.pickle", "rb") as f:
    subj = pickle.load(f)          # dict, len 165
trial = subj[0]                    # the first sentence (sentenceId == 1)
```

Every trial has at least:

| key | type | meaning |
|-----|------|---------|
| `sentenceId` | int (1..165) | 1-based sentence index, the stable join key |
| `blockIndex`, `block` | int | recording block bookkeeping |
| `persian_sentence` | str | the Farsi sentence shown to the subject |
| `word` | dict | per-word data, keyed by word position `0..n-1` (see §5) |

The **full** pickle adds the fields in §4; the **thin** `TRT_Total` pickle stops here and
its `word` entries carry only `{Id, content, nFixations, TRT_total}`.

---

## 4. Full pickle — extra trial-level fields

(Only in `task1_with_total/<name>/task1.pickle`.)

| key | shape | meaning |
|-----|-------|---------|
| `english_sentence` | str | original English sentence (Persian is a translation) |
| `raw_data` | `(126, T)` float32 | continuous **reading-period** EEG, channels × time (1200 Hz). `T` varies per sentence (e.g. 6440 ≈ 5.4 s). |
| `mean_{band}` | `(1, 126)` | sentence-mean band power per channel (see §6 for bands) |
| `mean_{band}_sec` | `(6, 126)` | time-binned version (6 segments) of the above |
| `mean_{band}_diff` | `(1, 59)` | left/right hemisphere power **difference** (59 electrode pairs) |
| `mean_{band}_diff_sec` | `(6, 59)` | time-binned hemisphere difference |
| `allFixations` | dict | every fixation in the trial: `duration`, `x`, `y`, `pupilSize`, each a 1-D array of equal length |
| `omissionRate` | float | fraction of words that received no fixation |
| `answer_data` | `(T, 126)` for control / `(2,)` stub otherwise | **answer-period** EEG (only the 15 control trials had a question to answer). Note: time × channels, i.e. transposed vs `raw_data`. |
| `answer_mean_{band}*` | `(2,)` stub or populated | answer-period analogues of the `mean_*` features |

> Important (verified across all 12 subjects × 15 control trials): reading-period
> `raw_data` `(126,T)` is present for **every** trial, controls included — it is real EEG
> (realistic ±50 µV amplitudes, length tracks reading time) and is a *different* array
> from `answer_data` (not a transpose/copy). The thing that exists **only** for the 15
> control trials is `answer_data` `(T,126)` = the EEG recorded while the subject answered
> the comprehension question.
>
> Note on a common confusion: *derived* datasets in this project drop controls. The
> `Thesis-Data Structures.ipynb` preprocessing does `if is_control(idx): continue`, so any
> file built from it (e.g. `padded_eeg`, `eeg_list`) contains **no control EEG**. If you
> looked at one of those and concluded "controls have no reading EEG / only answer EEG",
> that's the derived file's doing — the **raw pickles do contain control reading EEG**.
>
> Caveat: the **current** pickles in this folder contain control reading EEG; **earlier
> versions may not have** — if it matters for your work, verify in your own copy with the
> inspector in §12.
>
> Be mindful anyway: the dataset author (Parmis) recalls control trials lacking proper
> reading EEG and skipped them for that reason in earlier work, even though the current
> pickles do contain it. The reason isn't recorded in the preprocessing code (the skip is
> an unconditional `if is_control(idx): continue`). So before relying on control-trial EEG,
> sanity-check the actual values you're using (§12) rather than assuming they're valid.

---

## 5. Word-level structure

`trial['word']` is a dict keyed by word position `0,1,2,...`. Each entry:

**Thin (`TRT_Total`) pickle** — minimal:

| key | meaning |
|-----|---------|
| `Id` | word index (== the dict key) |
| `content` | the Persian word string |
| `nFixations` | number of fixations on this word |
| `TRT_total` | **full-band** EEG over the word's total reading time — `(126,)` float if the word was fixated, else a `(2,)` uint64 **stub** (treat as zeros) |

**Full pickle** — minimal fields above plus the complete feature family:

- `fixPositions` `(2,)`, `meanPupilSize`, `rawEEG` (list), `rawET` (list)
- For each **eye-tracking measure** `M ∈ {FFD, TRT, GD, GPT, SFD}` (see §7):
  - `M_{band}` — band power over that measure's time window, `(126,)` if fixated else `(2,)` stub
  - `M_{band}_diff` — `(59,)` left/right hemisphere difference
  - `M_total` — full-band version, `(126,)` or `(2,)` stub
  - `M_total_diff` — `(59,)`
  - `M`, `M_pupilsize` — often `None`

So per word there are 5 measures × 8 sub-bands (+ totals) × (value & hemisphere-diff).
The model in this project uses only `TRT_total`.

### The `(2,)` stub convention (critical)
When a word was **not** fixated (`nFixations == 0`) or a feature wasn't computable, the
array is a length-2 `uint64` placeholder instead of a real `(126,)`/`(59,)` vector.
Always check shape before use:

```python
import numpy as np
v = np.asarray(word['TRT_total'])
vec = v if v.shape == (126,) else np.zeros(126, dtype=np.float32)
```

---

## 6. Frequency bands

The data stores **8 sub-bands**; the thesis reports **4 bands** (each = a pair):

| sub-band keys | thesis band | range (Hz) |
|---------------|-------------|-----------|
| `t1` (4–6), `t2` (6.5–8) | **theta** | 4–8 |
| `a1` (8.5–10), `a2` (10.5–13) | **alpha** | 8.5–13 |
| `b1` (13.5–18), `b2` (18.5–30) | **beta** | 13.5–30 |
| `g1` (30.5–40), `g2` (40–49.5) | **gamma** | 30.5–49.5 |

`*_total` is the full band (0.5–50 Hz). To reproduce a 4-band feature you average/stack
the two matching sub-bands (exact collapse used by the original BiLSTM notebooks is still
being confirmed).

`126` = EEG channels. `59` = number of left/right electrode **pairs** used for the
hemisphere-difference (`_diff`) features.

---

## 7. Eye-tracking measures (the `M` in §5)

Standard reading measures, each defining a time window over which EEG band power is
averaged:

- **FFD** — first fixation duration (duration of the first fixation on the word)
- **SFD** — single fixation duration (only if the word was fixated exactly once)
- **GD** — gaze duration (sum of fixations on first pass, before leaving the word)
- **GPT** — go-past time (first entering the word until moving past it to the right)
- **TRT** — total reading time (sum of *all* fixations on the word, including re-reading)

---

## 8. Labels file

`teco_sentiment_labels_task1.csv` — 165 rows:

| column | notes |
|--------|-------|
| `sentence_id` | the **ZuCo / SST sentence id** (range 5..403) — NOT the TeCo 1..165 id. Do not use it to index the pickles. |
| `sentence` | Persian sentence (matches `persian_sentence` in the pickles) |
| `sentence_en` | English original |
| `sentiment_label` | **-1 / 0 / 1** = negative / neutral / positive |
| `control` | `CONTROL` for the 15 control sentences, else empty |

**Because the CSV id does not match the pickle id, join labels to EEG by sentence text**,
not by id. Normalizing (strip whitespace + remove the zero-width non-joiner `‌`/U+200C)
gives a 165/165 exact match. The same labels also live in the original
`sentences/Task_1/Total_Sentiment_Raw_*.xlsx` (there the `id` column *is* 1..165, and a
`sentence_id` column holds the ZuCo/SST id).

Why the id is a ZuCo id: TeCo's Task-1 sentences are **exactly ZuCo 1.0's sentiment
sentences** (both come from SST). This was verified by aligning the two corpora — every
TeCo Task-1 English sentence matches a ZuCo 1.0 sentence — so the `sentence_id` here is the
shared ZuCo/SST identifier, which is what makes English↔Persian (ZuCo↔TeCo) comparison
possible.

---

## 9. Minimal recipes

**Load full-band TRT EEG for one subject as a padded (sentence, word, channel) grid:**

```python
import pickle, numpy as np

with open("Amin_trt_total.pickle", "rb") as f:
    subj = pickle.load(f)

MAXW = max(len(subj[t]['word']) for t in subj)        # 49 for task 1
grid = np.zeros((165, MAXW, 126), dtype=np.float32)
for t in subj:
    sid = subj[t]['sentenceId']
    for wi, w in subj[t]['word'].items():
        v = np.asarray(w['TRT_total'])
        if v.shape == (126,):
            grid[sid - 1, wi] = v                     # word placed at its Id; rest stay 0
```

**Attach binary sentiment labels (drop neutral, keep controls → 110 sentences):**

```python
import pandas as pd, re
def norm(s): return re.sub(r"\s+", "", str(s)).replace("‌", "")

lab = pd.read_csv("teco_sentiment_labels_task1.csv")
label_of = {norm(r.sentence): r.sentiment_label for r in lab.itertuples()}
text = [None]*165
for t in subj: text[subj[t]['sentenceId']-1] = norm(subj[t]['persian_sentence'])

y = np.array([label_of[k] for k in text])             # {-1,0,1}, in sentence order
keep = np.where(y != 0)[0]
X, y = grid[keep], (y[keep] == 1).astype(int)         # neg->0, pos->1
```

**Pull a frequency band (full pickle), e.g. theta total reading time:**

```python
# word['TRT_t1'] and word['TRT_t2'] are (126,) when fixated; average the pair for theta
```

---

## 10. Gotchas checklist

- Trials are keyed `0..164`; `trial_key = sentenceId - 1`.
- CSV `sentence_id` ≠ pickle `sentenceId`. Join on sentence **text**.
- `(2,)`-shaped arrays are **stubs** (no fixation / not computed) → treat as zeros.
- Two pickle name spellings differ (`Razieh`/`razie`, `Shahrzad`/`sharzad`, `Amin`/`armin`).
- `raw_data` is `(126, T)` (chan×time); `answer_data` is `(T, 126)` (time×chan) — transposed.
- Binary sentiment set = 110 (controls kept). Skipping controls → 100.
- 12 subjects in `TRT_Total`; full `task1_with_total` now also has all 12.
- Full pickles are ~2 GB but load in <1 s; arrays are small except `raw_data`/`answer_data`.

---

## 11. ZuCo (for context)

**ZuCo** (Zurich Cognitive Language Processing Corpus; Hollenstein et al., 2018 ZuCo 1.0
and 2019 ZuCo 2.0) is the **English** simultaneous EEG + eye-tracking reading dataset that
TeCo is modelled on. It covers the same kinds of tasks (normal reading, sentiment on movie
reviews, and Wikipedia relation extraction) and the **same eye-tracking measures**
(FFD, TRT, GD, GPT, SFD) and EEG band features — so TeCo's per-word feature naming mirrors
ZuCo's by design, which is what makes cross-lingual comparison possible.

Differences to keep in mind:
- ZuCo is distributed as **MATLAB `.mat`** files (often `h5py`/`scipy.io` to read), not
  pickles; its electrode montage and sampling differ from TeCo's.
- ZuCo is **not included in this `thesis_data` folder**. If you see ZuCo referenced in
  Parmis's other experiments, it comes from the public ZuCo release / a separately
  prepared `padded_eeg` of raw waveforms, not from these TeCo pickles.

When a task says "compare with Hollenstein/ZuCo", that means the published English numbers
— the structures above are TeCo (Persian) and are self-contained.

**ZuCo missing-data note.** Unlike TeCo, ZuCo genuinely has missing/NaN trials and words
(e.g. subject `DN`), so the ZuCo processing here is full of `is_nan()` guards on `rawData`,
`word`, `nFixations`, and band values. This is *not* tied to control sentences — it's
generic NaN handling. (If you remember "control trials had no EEG", that intuition most
likely comes from ZuCo's NaN trials and/or TeCo's deliberate control-skipping, not from a
missing-EEG flag on TeCo controls.)

---

## 12. Quick inspector

Drop this in to print the schema of any TeCo pickle and verify the structure (and the
control-EEG question) in your own copy:

```python
import pickle, numpy as np

def inspect(path, n_show=6):
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"file: {path}")
    print(f"trials: {len(d)}  (keys {min(d)}..{max(d)})")
    tr = d[0]
    print("\ntrial[0] keys:")
    for k, v in list(tr.items())[:n_show]:
        if isinstance(v, np.ndarray): print(f"  {k:18s} ndarray{v.shape} {v.dtype}")
        elif isinstance(v, dict):     print(f"  {k:18s} dict(len={len(v)})")
        else:                         print(f"  {k:18s} {type(v).__name__}: {repr(v)[:45]}")
    if len(tr) > n_show: print(f"  ... (+{len(tr)-n_show} more)")
    w = tr['word'][0]
    print("\ntrial[0]['word'][0] keys:")
    for k, v in list(w.items())[:n_show]:
        if isinstance(v, np.ndarray): print(f"  {k:14s} ndarray{v.shape} {v.dtype}")
        else:                         print(f"  {k:14s} {type(v).__name__}: {repr(v)[:40]}")
    if len(w) > n_show: print(f"  ... (+{len(w)-n_show} more)")

inspect("TRT_Total/Yaser_trt_total.pickle")
```

Output on the **thin** `TRT_Total` pickle:

```
file: TRT_Total/Yaser_trt_total.pickle
trials: 165  (keys 0..164)

trial[0] keys:
  sentenceId         int: 1
  blockIndex         int: 1
  block              int: 1
  persian_sentence   str: 'واقعاً فیلمی تاثیرگذار است چرا که به طرز واق
  word               dict(len=14)

trial[0]['word'][0] keys:
  Id             int: 0
  content        str: 'واقعاً'
  nFixations     int: 0
  TRT_total      ndarray(2,) uint64
```

On the **full** `task1_with_total/<name>/task1.pickle` the same call prints the much larger
field set from §4–5 (`raw_data (126,T)`, `mean_{band}*`, `allFixations`, `answer_data`, and
per word the full `FFD/TRT/GD/GPT/SFD × bands` family). To check control reading EEG
directly:

```python
import numpy as np
controls = {8,11,17,29,38,39,41,59,64,81,92,96,117,148,150}   # sentenceIds
d = pickle.load(open("TeCo/task1_with_total/yaser/task1.pickle", "rb"))
for t in d:
    if d[t]['sentenceId'] in controls:
        rd = np.asarray(d[t]['raw_data'])
        print(d[t]['sentenceId'], "raw_data", rd.shape, "real" if rd.ndim == 2 else "STUB")
```

