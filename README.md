# Stem4Health-hackathon
STEM4Health Hackathon - with Johnson &amp; Johnson

# Task
Build a strategy for process performance monitoring models that stay robust, adapt to new conditions, and learn from limited data, with the potential to leverage self-supervised learning techniques. 
In industry, the real challenge is usually harder. Rejected cases are rare, sometimes not labelled at all, and can take forms never seen before. In practice, the goal is to detect anomalies (the equivalent of rejected cases) by learning only from accepted cases.
We were encouraged to build a second solution along these lines: train only on accepted instances and flag anything that deviates from normal behaviour. This solution is not scored on the leaderboard, but we could include it in our repository and present it in the final presentation. Approaches that work well here are very valuable for real-world use. 

# Data
Each process comes as a calibration set with labels, and a test set without labels. 
Batches are time series. A batch is one production run with its own ID (for example A_003). There are no timestamps, but the rows are in order within each batch, so we created a time step with t = groupby("batch_id").cumcount() (0, 1, 2, … restarting for each batch). The team later moved loading into a shared data_loader.py (in src/), whose load_long() reads a fast parquet export and adds t automatically.
Merging labels. The labels were joined on batch_id with a left merge; every row received a label. 
Borderline relabelling. The labels file marks A_101 and A_103 as accepted, while the brief says A_101–A_104 are unacceptable. We followed the brief and relabelled them as rejected. A_102 only appears in the test data.
Outliers and shutdowns. describe() showed minimum values far below the normal range: flow_rate_1 drops to −19 against a typical 130.9–131.5, flow_rate_2 to 347 against about 1,847, pressure_1 to 2 against about 24.7. pressure_2 and pressure_4 go negative and spike to 70–80. These look like start-ups, shutdowns or sensor glitches. We decided not to remove outliers from the whole dataset, because in anomaly detection outliers can be the signal. Cleaning, if any, applies only to training data and only after checking whether the extreme rows belong to rejected batches.

# Strategy
We wanted to see what a normal sample looked like from the accepted batches, and then flag anything too far from it. We used an anomaly detection algorithm, instead of a classification algorithm. 

Validation: leave one batch out
We validate by leaving out whole batches, never single rows. Rows from the same batch are strongly correlated, so splitting rows would let the model recognise a run it has partly seen and make results look better than they are.
The loop, used for every model:
• Each accepted batch is left out once. A model is trained on the remaining accepted batches and scores the left-out one. This measures false alarms on unseen good batches.
• Rejected batches never enter training. They are scored by every fold's model and the scores are averaged. This measures missed rejects.
• All scores are collected and the threshold is chosen once, across all folds.
With 161 batches this exact leave-one-out is fast. The LSTM was the exception: it used 5-fold cross-validation over accepted batches, because training 121 networks would take too long. 

# Features
The final model describes each batch with 187 numbers for process A: 55 base features plus 132 extra features.
Base features (55)
For each of the 11 sensors, over the whole batch: mean, standard deviation, minimum, maximum and slope (linear trend over t). The slope was first computed with np.polyfit per batch and then replaced by a vectorised formula, slope = cov(t, x) / var(t), which is much faster.
Extra features (132)
Whole-batch statistics average away anything brief, phase-specific or relational. The extra features target exactly that: 
• Phase means: Each sensor's mean at the start (first 20%), middle and end (last 20%) of the batch
• 5th / 95th percentiles: Typical low and high, robust to single glitches
• Roughness Mean absolute: step-to-step change: how jumpy a signal is
• Worst local instability: Maximum of a 50-step rolling standard deviation
• Sensor-pair correlations: Whether sensors still move together as usual

For process B, with 20 sensors, the same functions produce more features, mainly from the 190 sensor pairs.
 
# Models and results
All generic anomaly detectors landed at about the same quality, while PCA Hotelling's T² caught clearly more rejected batches. Model choice mattered less than how the anomaly score is built.

Models compared
Every model was wrapped to the same interface (fit on accepted batches, return an anomaly score) and run through the same leave-one-batch-out loop. Features were standardised inside each fold.
• Isolation Forest: the first baseline.
• Local outlier factor (LOF), with novelty=True so it can score unseen batches.
• Robust covariance (elliptic envelope), after PCA to 10 components, because 120 batches are too few for a stable 55-feature covariance.
• One-class SVM (RBF kernel) and one-class SVM with SGD (Nyström kernel approximation, mainly for completeness).
• K-means: distance to the nearest of 3 cluster centres.
• Angle-based outlier detection (ABOD): a custom fast version on 10 nearest neighbours.
• LSTM autoencoder on the raw time series, resampled to 100 steps; anomaly score = reconstruction error.
• PCA T², PCA Q and T² + Q: classic multivariate statistical process control.
• Top-3 robust z-score: the mean of a batch's three most extreme features.
• Rank ensembles of several models. 

All models other than T² had ROC AUC between 0.80 and 0.87, which is within noise with only 40 rejects. PCA T² reached ROC AUC 0.866, PR AUC 0.677 and MCC 0.55, against MCC 0.37–0.44 for the rest, and it caught the one borderline batch present. The supervised logistic-regression baseline scored ROC AUC 0.66, and the first rank ensemble improved PR AUC only slightly (0.674). 

What the diagnostics showed
Comparing caught and missed rejects (z-scores against accepted batches) explained the gap:
• The apparent "density drift" failure mode came from one single batch with an extreme density slope; it dragged the average. Lesson: compare medians, not means.
• About 10 missed rejects had temperature_1 std of 0.26–0.33, above every accepted batch. One feature separated them perfectly, but generic detectors diluted that deviation across 55 features. Robust covariance also dropped it by keeping only the 10 directions where good batches vary most.
• The remaining missed rejects sit inside the accepted cloud and cannot be told apart on these features.
T² measures distance in units of normal variation, so a small deviation in a direction where good batches barely move counts heavily. That is why it caught the unstable-temperature group.

Final configuration
After relabelling the borderline batches (119 accepted, 42 rejected), adding the extra features and raising PCA to 15 components, the chosen model was T² on the 187 extended features: standardise → PCA (15 components) → Hotelling's T², scaled by the median training score. Its leave-one-batch-out F1 at the chosen threshold was 0.897.

# Metrics and the alarm threshold
The leaderboard scores the average of the F1 scores for A and B, so we tune the threshold to maximise F1 on process A.

Metrics used
In our tables the positive class is rejected. All threshold metrics come from four counts: TP (bad batch caught), FN (bad batch missed), FP (false alarm) and TN (good batch left alone). 

Recall (sensitivity): TP / (TP + FN): Share of bad batches caught
Miss rate: FN / (TP + FN): Share of bad batches missed
Specificity: TN / (TN + FP): Share of good batches left alone
False alarm rate: FP / (FP + TN): How often operators are bothered for nothing
Precision: TP / (TP + FP): Share of alarms that were real
Accuracy: (TP + TN) / all: Misleading here: "accept everything" scores 0.75
Balanced accuracy: (recall + specificity) / 2: Fair under imbalance; guessing = 0.5
F1: Harmonic mean of precision and recall
F2: Like F1, recall weighted twice: Matches "misses are worse"
MCC: Correlation of predictions with truth, −1 to 1: Fairest single number under imbalance
ROC AUC: Chance a random reject scores above a random good batch: Threshold-free; 0.5 = guessing
PR AUC: Area under precision–recall curve: Threshold-free; baseline ≈ 0.25 here

Lowering the threshold catches more rejects but raises false alarms. No threshold maximises everything; only better separation (higher ROC/PR AUC) improves all metrics at once.

Reading the test result
On the first test submission, process A gave accuracy 0.733, precision 0.829, recall 0.808 and F1 0.819. Accuracy below both precision and recall is only possible if the positive class makes up about 74% of test batches. That matches the share of accepted batches, so the leaderboard most likely treats accepted as positive. Under that reading the model caught about 51% of test rejects (close to 52% on calibration) with about 19% false alarms.

Choosing the threshold
The threshold is the value on the leave-one-batch-out scores that maximises F1 with accepted as positive, reweighting the classes to the test balance. Under that reading the best F1 for the generic models was 0.87–0.90; under the opposite reading it was 0.89–0.91, with opposite thresholds. The final model's threshold is 3.91: it flags 5% of accepted calibration batches. For the final submission it is applied as an absolute value to the final model's scores, not recomputed as a quantile, because in-sample scores run lower than out-of-fold ones.
These F1 values are somewhat optimistic, since the threshold was chosen on the same scores. Trivial baselines are also high: "accept everything" gives F1 ≈ 0.85 at a 74/26 split.

# Test predictions and the final pipeline
The final model flags 16% of process A's test batches as rejected, which matches what calibration predicts if about 26% of test batches are rejects.

Applying the model to the test data
The test files have no labels, so no cross-validation or metrics run on them. Instead:
1.  Build the same 187 features for the test batches, with the same functions and column order as calibration.
2.  Fit one final model on all accepted calibration batches.
3.  Score the test batches.
4.  Flag every batch above threshold 3.91 as rejected and save the predictions.

Sanity checks on process A, done without labels:
• No missing values in the 161 × 187 test feature table.
• Flag rate. Calibration predicts 0.74 × 5% + 0.26 × ~50% ≈ 17% flagged; the test gave 16% (15.5% rejected, 84.5% accepted). The test data behaves as the model expects, with no large shift.
• Score distributions. Scores are heavy-tailed (some reach the thousands), so test and calibration histograms are compared on a log scale.
• Batch IDs. The test set has 161 batches, the same as calibration, so we added a check for overlap with calibration IDs and for A_102, which should appear only in the test data.

The final notebook
final_model.ipynb contains only what the final model needs. It sits in notebooks/ next to inspecting.ipynb and finds the team's data_loader.py in src/ automatically. Run All:
1.  Imports and settings: N_COMPONENTS = 15, POSITIVE = "accepted", borderline batch list.
2.  The PCAMonitor class: standardise → PCA → Hotelling's T² (Q and T² + Q also available).
3.  Feature functions: base_features, extra_features, make_features.
4.  Labels with borderline relabelling, leave-one-batch-out scoring, F1-based threshold search.
5.  run(process): features, validation, threshold, final fit, test predictions, submission_<process>.csv.
6.  Runs process A and B.
7.  Concatenates both into submission.csv with columns batch_id and label.
Exploration (plots, the other models, metric tables, ensembles) stays in inspecting.ipynb.

# Limitations, open questions and pitch points
The biggest open question is which class the leaderboard counts as positive, because it reverses the direction of threshold tuning.

Limitations
• Small number of rejects. With 40–42 rejects, differences of about 0.02 in AUC are noise. We report counts ("21 of 40") alongside percentages.
• Optimistic calibration F1. The threshold and the model were selected on the same leave-one-batch-out scores.
• An information ceiling. Some rejects sit inside the accepted cloud. If they were rejected for lab results the sensors don't reflect, no model can find them.
• Borderline batches. Only two or three appear in calibration; they score just below the alarm line, as expected for batches just outside acceptability.
• Process B. The perfect 1.0 should be checked for leakage (scaler or threshold fitted on test data); otherwise B is simply an easier simulated process.

Open questions for the organisers
[ ] Which class is positive in the scoring, and is the score the plain average of the two F1s?
[ ] How much worse is a missed bad batch than a false alarm (the cost ratio)?
[ ] Is temperature_1 (about 14 °C) an operating condition, such as inlet or cooling water? If so, a warm batch may be acceptable variability rather than a fault.

Ideas not yet finished
• Trajectory models (multiway PCA on batches resampled to 100 steps) that compare each batch with the normal profile at the same point in time.
• Ensembles of models that see different views: summary features, trajectories, LSTM.
• A paired bootstrap on PR AUC to confirm that an improvement is real before switching models.
• A three-tier output (OK / novel, review / reject) separating new-but-acceptable variability from failures.

Pitch points
• One strategy, two processes: learn normal from accepted batches, score distance from normal, set the alarm by cost. Only the features are process-specific.
• Why anomaly detection: the supervised baseline (ROC AUC 0.66) lost to every anomaly detector, because rejects come from several different failure modes.
• Why T²: eight generic detectors tied at ROC AUC ≈ 0.85; T² caught 21 of 40 against 12–16 for the rest. It is the established method in pharma process monitoring, explainable and fast.
• What the diagnostics revealed: a group of rejects with unstable temperature_1 that generic models diluted away, and a failure mode that turned out to be a single batch.
• Honest metrics: accuracy is misleading at 75% accepted; we report recall, false-alarm rate, F1/F2 and MCC, and state the threshold trade-off as the answer to "how much variability can we accept?"
The model we chose was the Hotelling’s T^2 test, and we added extra features. 
