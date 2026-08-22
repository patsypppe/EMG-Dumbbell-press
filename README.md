# EMG Exercise Form Classification

**Turns raw two-channel surface EMG from a dumbbell press into a Good / Fair / Bad form label, using the
relationship between the deltoid and pectoral channels rather than either channel alone.**

Research code behind a peer-reviewed conference paper.

> **Status: research notebooks, released as-is.** The two notebooks are the ones used for the
> experiments, with their cell outputs committed so results are readable without re-running. Best
> validation accuracy recorded in those outputs is **92.5%**, on a validation split of roughly 80
> samples. That sample size is small enough that the figure should be read as indicative. See
> [Current limitations](#current-limitations).

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
thresholded into `Good`, `Fair`, and `Bad`. This is a heuristic label, not an expert annotation, which
matters when reading the accuracy figure: the model is learning to reproduce a threshold rule applied to
features derived from the same signal.

**Models.** Keras `Sequential` stacks of LSTM and SimpleRNN layers with batch normalization and dropout,
into a 3-way softmax. Trained for 10 to 30 epochs at batch size 32 with a 0.2 validation split.

Note that these are **recurrent and dense models, not convolutional**. There is no `Conv1D` or `Conv2D`
anywhere in either notebook.

---

## Results

Taken from the committed cell outputs in `EMG.ipynb`, not from memory:

| Metric | Value |
|---|---|
| Best validation accuracy | **0.9250** |
| Second-best validation accuracy | 0.9125 |
| Best training accuracy | 0.9054 |
| Approximate validation set size | ~80 samples |

The validation accuracies quantize on 1/80, which is how the validation set size above was inferred.
Read the headline number with that in mind: a single sample moves it by 1.25 points.

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

Run the cells top to bottom. `EMG.ipynb` writes two intermediate files into the working directory
(`GYM_ATTRIBUTES_DATASET.csv` and `preprocessed.csv`) which later cells read back, so cells must be run
in order.

---

## Current limitations

- **Small validation set.** Roughly 80 samples. Every reported accuracy should be read as indicative
  rather than precise.
- **The labels are heuristic, not annotated.** They come from thresholding a min-max scaled feature
  derived from the same signal the model sees. The task is therefore closer to reproducing a rule than
  to learning form from ground truth.
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
