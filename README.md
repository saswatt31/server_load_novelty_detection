# Server Load Novelty Detection (One-Class SVM) — PRIORITY

Detects **unexpected CPU usage spikes** and **memory leaks** in Node.js-based
automation containers using a **One-Class SVM** trained only on healthy telemetry.

## Why One-Class SVM?
Anomaly/novelty detection is used because container failures are rare and
unlabelled: we only have plenty of "normal" metrics. The One-Class SVM learns a
boundary around healthy operating points (RBF kernel) and flags anything far
outside it as a *novelty* (label `-1`), while in-distribution samples get `+1`.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Simulates container telemetry (CPU %, Mem %, load, net): healthy daily cycle + injected CPU-spike and memory-leak incidents. Writes `data/metrics.csv` |
| `detect_anomalies.py` | Builds sliding-window features, fits One-Class SVM, scores every window, exports `results/anomaly_report.csv` |
| `evaluate.py` | Since incidents are synthetically injected, true labels exist: computes precision / recall / F1 / ROC-AUC and the confusion matrix |
| `plot_results.py` | Timeline plot: raw metrics with detected anomalies highlighted + decision-score over time |
| `compare_baselines.py` | Tuned baseline comparison (3-sigma rules, Isolation Forest vs One-Class SVM), per-incident-type recall, detection latency, error profile, post-leak label-artefact analysis, comparison charts |
| `make_report.py` | Builds `Project_Report_Server_Load_Novelty_Detection.docx` by filling `Project_Report_Template.docx` |
| `check_report_template.py` | Verifies the report against the template (sections, tables, artefacts, length) -> `report_template_compliance.txt` |
| `make_slides.py` | Builds the 10-slide presentation `Presentation_Server_Load_Novelty_Detection.pptx` (python-pptx, 16:9) |
| `check_slides_layout.py` | Layout audit of the deck: text overflow, text collisions, off-slide shapes |
| `render_slides_preview.py` | Renders the deck to PNGs + a self-contained HTML gallery (`slides_preview/`) |

## How to run
```bash
python generate_data.py       # -> data/metrics.csv
python detect_anomalies.py    # -> results/anomaly_report.csv (SVM training + scoring)
python evaluate.py            # -> metrics + confusion matrix
python plot_results.py        # -> results/timeline.png
python compare_baselines.py   # -> tuned baselines, latency, error analysis, charts
```
Or everything at once: `python run_all.py`

## Baseline comparison (tuned operating points, same windows)
| Detector | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| One-Class SVM (default, nu = 0.05) | 0.954 | 0.697 | 0.991 | 0.818 | 0.992 |
| Static 3-sigma rule (k = 4) | 0.993 | 0.955 | 0.980 | 0.967 | 0.994 |
| Isolation Forest (contamination = 0.02) | 0.979 | 0.849 | 0.969 | 0.905 | 0.987 |
| One-Class SVM (tuned, nu = 0.005) | 0.990 | 0.927 | 0.983 | 0.954 | 0.994 |

The 3-sigma rule stays marginally ahead on F1 because it is calibrated per
feature on the healthy data and designed around exactly the injected failure
signatures; the One-Class SVM needs no per-metric thresholds, has slightly
higher recall and is clearly stronger than Isolation Forest. 208 of the 323
default false positives (64.4%) fall in the post-leak regime where the simulator
leaves memory elevated (70% vs 56.7% healthy) but labels only the ramp.

## Project report
`Project_Report_Server_Load_Novelty_Detection.docx` follows the supplied
`Project_Report_Template.docx` section for section (all 7 sections, 39 sub-
headings, five tables, three figures, a code snippet and five references).
Regenerate and re-verify with:
```bash
python make_report.py            # rebuild the DOCX from the template
python check_report_template.py  # compare report vs template -> PASS/FAIL
```
The submission details are already filled in from the project constants at the
top of `make_report.py` (edit them there and re-run to regenerate):

* Group 4 - 7th Semester, B.Tech Artificial Intelligence and Machine Learning
  (AIML), Academic Year 2023-2027
* Saswat Pattanaik (FET-BAML-2023-27-026), Ritesh Kumar Singh
  (FET-BAML-2023-27-027), Manoranjan Barik (FET-BAML-2023-27-039),
  Ankit Kumar Das (FET-BAML-2023-27-018)
* Guide: Mr. Rahul Rawat | Faculty of Engineering and Technology (FET)
* Subject: Pattern Recognition and Anomaly Detection

Only `Place` and `Date` are left blank on the declaration page for hand
signing. In Word, right-click the table of contents and choose "Update Field"
to fill in the page numbers.

## Presentation (10 slides)
`Presentation_Server_Load_Novelty_Detection.pptx` - a 16:9 dark-themed deck
built from the project's real figures, with speaker notes on every slide:

| # | Slide | Content |
|---|---|---|
| 1 | Introduction | title, one-line scope, technology chips, group details |
| 2 | The problem | CPU spikes and memory leaks, why fixed thresholds fail |
| 3 | The idea | supervised vs one-class novelty detection |
| 4 | Data | telemetry simulator, 7,200 samples, 4 injected incidents |
| 5 | Features | 6-minute window, 16 level/spread/range/slope features |
| 6 | Model | RBF One-Class SVM pipeline, nu as the sensitivity dial |
| 7 | Results | recall 0.991, precision 0.697, F1 0.818, ROC-AUC 0.992 + timeline |
| 8 | Benchmark | tuned baselines vs proposed detector + verdict card |
| 9 | Insight | confusion matrix, error analysis, nu sensitivity sweep |
| 10 | Thank you | questions and discussion |

```bash
python make_slides.py             # rebuild the .pptx (needs python-pptx)
python check_slides_layout.py     # layout audit: overflow / collisions / bounds
python render_slides_preview.py   # PNG gallery -> slides_preview/index.html
```
The submission details on slides 1 and 10 come from the constants at the top of
`make_slides.py` (kept in sync with `make_report.py`). The deck keeps the same
honest framing as the report: the tuned 3-sigma rule is about one F1 point
ahead, while the One-Class SVM needs no per-metric thresholds, matches ROC-AUC
and clearly beats Isolation Forest.

## Expected output (verified run)
```
trained on 6,438 healthy windows (16 features, nu=0.05, gamma=scale)
scored 7,188 windows -> 1066 flagged as novelty (14.8%)
Confusion matrix (rows=truth, cols=pred):
              normal  novel
   normal       6115    323
   novel           7    743
Precision: 0.6970  Recall: 0.9907  F1: 0.8183  ROC-AUC: 0.9915
```
Recall is the key figure — we catch ~99% of incidents (only 7 missed windows
out of 750). The false-positive rate is 5.02% of healthy windows, which matches
the 5% tolerance requested by `nu`; lowering `nu` to 0.005 raises precision to
0.927 for a loss of only 0.8 points of recall.

## Tuning knobs
- `NU` (0.05) — upper bound on the fraction of training outliers / lower bound on support vectors.
- `GAMMA` (`scale`) — RBF width; larger = tighter, more sensitive boundary.
- `WINDOW = 12` — 12 × 30 s = 6-minute context per feature vector.
- `TRAIN_HOURS` — restrict training to known-healthy hours if you have them.
