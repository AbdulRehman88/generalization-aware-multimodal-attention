# Deep-Learning Scientific Results Audit v1

Source hashes:

- operation_results.csv: `3bb1454cc7fc63476e32a43610323cdde9563029e7e46ececc739874e824d611`
- internal_nested_loso_selected_durations.csv: `ffe11e6f358c305cd780b836e35bbf6fff10927f2bbae922398c74bbba5eba79`

Ranking rule: mean test balanced accuracy first, mean test macro-F1 second. No best-seed selection is used.

## Strict nested LOSO

Best mean strict-LOSO result: **ShallowConvNet / EEG**, balanced accuracy 0.5764  0.1059, macro-F1 0.5220  0.1321 across 39 fold-seed selected evaluations.

Duration selection counts are reported separately; Pupil-containing TCN paths were structurally fixed to 8 s before performance was observed.

## Subject-adaptive

- 30 s/class: best mean result **ShallowConvNet / EEG**, balanced accuracy 0.6464  0.1002, macro-F1 0.6191  0.1342.
- 60 s/class: best mean result **ShallowConvNet / EEG**, balanced accuracy 0.6388  0.1336, macro-F1 0.5962  0.1755.
- 120 s/class: best mean result **ShallowConvNet / EEG**, balanced accuracy 0.7154  0.1372, macro-F1 0.6885  0.1572.

## BBBD within-experiment

- experiment2: **MultibranchTCN / EEG**, balanced accuracy 0.5950  0.1588, macro-F1 0.4824  0.2243.
- experiment3: **MultibranchTCN / EEG_Pupil**, balanced accuracy 0.6562  0.1600, macro-F1 0.5931  0.2115.

## BBBD cross-experiment

- experiment2_to_experiment3: **MultibranchTCN / EEG**, balanced accuracy 0.6163  0.0235, macro-F1 0.6059  0.0300.
- experiment3_to_experiment2: **MultibranchTCN / Pupil**, balanced accuracy 0.6017  0.0126, macro-F1 0.5822  0.0033.

## Interpretation boundary

- Conventional internal protocols measure within-dataset capacity/comparability.
- Strict nested LOSO is the primary calibration-free unseen-subject internal evidence.
- Subject-adaptive results are a secondary personalization analysis.
- BBBD labels represent attentive-versus-distracted experimental conditions, not a pure latent attention construct.
- BBBD cross-experiment results are domain-shift evidence, not frozen transfer from the internal three-class task.
- No individual seed is selected for manuscript reporting.
- Mean and dispersion are retained so isolated high-performing runs are not presented as the main result.
