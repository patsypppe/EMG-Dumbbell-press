# EMG Exercise Form Classification

**Turns raw two-channel surface EMG from a dumbbell press into a Good / Fair / Bad form label, using the
relationship between the deltoid and pectoral channels rather than either channel alone.**

Research code behind a peer-reviewed conference paper.

> **Status: research notebooks, released as-is — and the accuracy figures in them do not measure
> what the title suggests.** The label is a threshold on a quantity that is also handed to the model
> as an input column, so the task the network is set is to recover a rule it can already see. A
> three-line `if` on that one column reproduces every label exactly. Read
> [What these numbers are not](#what-these-numbers-are-not) before quoting anything from here.

---

## The problem

Judging lifting form from EMG looks like a modelling problem and is mostly a signal problem. A dumbbell
press recruits the anterior deltoid and the pectoralis together, and bad form usually means the wrong
one is doing the work. Neither channel alone says much: a high deltoid reading is fine during part of
the movement and wrong during another. The informative quantity is the *balance* between them over a
window.

So the pipeline spends most of its effort on windowed feature extraction, and the model on top is
deliberately simple.

---

## Method

**Signal.** Two channels, deltoid and pectoral, sampled at 256 Hz over 123,072 samples.

**Windowed features**, computed per channel over a sliding window:

- root mean square, as an activation-magnitude proxy
- integrated area under the rectified signal, as total work in the window
- skewness and kurtosis, to capture burst shape rather than just amplitude
- Wilcoxon amplitude statistics between channels

**Label derivation.** `diff = pec_area - del_area` is min-max scaled, and the scaled difference is
thresholded into `Good`, `Fair`, and `Bad`. This is a heuristic label, not an expert annotation.

**And the quantity it is thresholded from stays in the feature matrix.** `X = areas_df.iloc[:, :-1]`
takes every column except the label, which means `scaled_difference` — and `diff`, and the two areas
that determine both — are all inputs. The target is a deterministic function of the input. This is
target leakage, and it is the single most important thing to know about every number below.

**Models.** Keras `Sequential` stacks of LSTM and SimpleRNN layers with batch normalization and dropout,
into a 3-way softmax. Trained for 10 to 30 epochs at batch size 32 with a 0.2 validation split.

Note that these are **recurrent and dense models, not convolutional**. There is no `Conv1D` or `Conv2D`
anywhere in either notebook.

---

## What these numbers are not

Taken from the committed cell outputs in `EMG.ipynb`, not from memory:

| Metric | Value |
|---|---|
| Highest validation accuracy recorded | 0.9250 |
| Lowest validation accuracy recorded | 0.4500 |
| Distinct `val_accuracy` values in the committed outputs | 10 |
| Approximate validation set size | ~80 samples |

Three reasons not to read 0.9250 as a result.

**A three-line rule beats it.** Because `scaled_difference` is an input column and the label is a
threshold on `scaled_difference`, the label can be recovered exactly without any model:

```python
rule = np.where((s > 0) & (s < 0.4), 'Bad',
       np.where((s >= 0.3) & (s < 0.5), 'Fair', 'Good'))   # s = X['scaled_difference']
# reproduces the labels: 100.00%  (586/586)
```

A network reported at 0.9250 is scoring *below* a rule it could have learned from one column.

**0.9250 is the maximum over ten recorded runs**, whose values run down to 0.4500. Reporting the
best of a spread that wide, on a validation set this small, is a selection effect rather than a
measurement.

**Train and test are not independent.** The sliding window is 6000 samples with a 200-sample hop,
so consecutive windows overlap by **96.7%**. `train_test_split(..., test_size=0.2, random_state=42)`
splits those overlapping windows at random, putting near-identical windows on both sides. Even with
the leakage removed, the held-out score would not measure generalization. A segment-wise or
subject-wise split is the fix.

The validation accuracies quantize on 1/80, which is how the validation set size was inferred. Running
the same windowing over the committed `EMG_DATASET.csv` yields 586 windows in total, which is
consistent with an inner validation split of that size.

---

## Files

| File | Contents |
|---|---|
| `EMG.ipynb` | Main pipeline: load, window, extract features, label, train, evaluate |
| `EMG_TEST.ipynb` | Shorter end-to-end pass from raw CSV through cleaning and visual inspection |
| `EMG_DATASET.csv` | Recorded signal, 123,072 rows |
| `PRANAV_256.txt` | The same recording in the acquisition tool's raw text format |

---

## Technology stack

Python, TensorFlow / Keras (LSTM, SimpleRNN, BatchNormalization, Dropout), pandas, NumPy,
SciPy (`skew`, `kurtosis`, `wilcoxon`), scikit-learn (`MinMaxScaler`, `train_test_split`),
Matplotlib, Seaborn.

---

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/patsypppe/EMG-Dumbbell-press.git
cd EMG-Dumbbell-press
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook EMG.ipynb
```

`EMG.ipynb` cell 0 reads `preprocessed.csv`, which is **not committed and is not written by any cell
in either notebook** — so a fresh clone cannot run the pipeline as-is. `EMG_DATASET.csv` holds the same
recording (time, deltoid, pectoral) and can stand in for it; the windowing in cell 0 then produces 586
windows.

---

## Current limitations

- **Target leakage.** The column the label is thresholded from is one of the model's inputs. Nothing
  reported here survives that, and it is not fixable by dropping one column: `diff` and the two area
  features determine the label just as completely.
- **There is no independent ground truth.** No one annotated these windows as good or bad form. The
  labels are a rule applied to the signal, so even a leakage-free model would only be learning the rule.
  Establishing whether EMG balance predicts form needs labels from outside the signal.
- **Overlapping windows, random split.** 96.7% overlap between consecutive windows, split at random.
- **Small validation set.** Roughly 80 samples. One sample moves the figure by 1.25 points.
- **`preprocessed.csv` is not in the repository.** `EMG.ipynb` cell 0 reads it and no cell in either
  notebook writes it, so a fresh clone cannot run the pipeline top to bottom as committed.
- **The threshold rule has overlapping branches.** In the labelling cell, the first condition
  `0 < i < 0.4` absorbs everything below 0.4, so the `Fair` branch `0.3 <= i < 0.5` can only ever fire
  between 0.4 and 0.5. The intended boundaries and the effective ones differ.
- **Single subject, single session.** The split is not subject-independent, so the numbers say nothing
  about generalization to a new person.
- **The two data files are the same recording** in two formats, roughly 8 MB committed twice.
- **Notebooks carry embedded output images**, which is why `EMG.ipynb` is around 780 KB.
- **A stray `load_iris` import** is left over from earlier scratch work.
- **The models are LSTM and SimpleRNN**, so any description of this work as CNN-based is incorrect.

---

## Citation

This work was published at a peer-reviewed IEEE conference. Citation details and DOI are available on
request or via [LinkedIn](https://www.linkedin.com/in/pranavpattanashetty).

---

## License

MIT. See [LICENSE](LICENSE).
