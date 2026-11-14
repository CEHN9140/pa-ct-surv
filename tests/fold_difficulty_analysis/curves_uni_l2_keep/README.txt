All curves use logged per-epoch values from the same ROI64 patient cohort and fixed five folds.
Settings: ABMIL, Cox batch 64, dropout 0.25, LR 1e-4, weight decay 5e-4, seed123, maximum 30 epochs, patience10.
Training C-index is online: predictions are collected in training mode while parameters change during the epoch. It is not a fixed-checkpoint eval-mode training C-index.
Training loss is the mean minibatch Cox loss; validation loss uses the entire validation cohort. Their numerical levels are not directly comparable because risk-set sizes differ.
No smoothing, interpolation, filling after early stopping, or averaging across folds.
Only training epochs 1 onward are plotted; unavailable epoch-zero values are not imputed.
KEEP features are 768-D with encoder output normalization; UNI features are 1024-D. KEEP is a different encoder, not a normalization-only control.
The selected best epoch marker uses maximum logged validation C-index; statistics tables use full-precision saved predictions.
