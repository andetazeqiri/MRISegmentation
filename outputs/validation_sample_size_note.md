Recommended thesis text — Validation set limitation and plan

Current limitation
- The held-out validation in our local comparisons is extremely small: n=4 BraTS 2020 patients (plus n=2 BraTS2021 in earlier checks). A paired Wilcoxon test performed on per-patient mean Dice returned p=0.875 (n=4), which is non-significant but uninformative due to the tiny sample.

Observed pilot effect
- Pilot per-patient differences (mean 2D minus 3D mean-regions Dice) = 0.8268, SD = 0.0328.
- This pilot shows a very large effect (Cohen's d ≈ 25) driven by our 2D model performing much better on these few cases — but the small n undermines generalization.

Recommended plan to strengthen inference
1. Expand the validation set to at least 30–60 independent patients to get robust, publishable paired comparisons.
   - Practical options: set `--val-count` to 55 or 60 when running the repo scripts to add +10/+15 patients to the validation split used previously.
2. Use patient-level mean Dice as the unit of analysis (paired test) and report:
   - The per-patient Dice table (CSV) and summary statistics (mean ± SD).
   - Paired Wilcoxon signed-rank test with exact p-value and effect size (paired Cohen's d or Hodges–Lehmann estimate).
3. For transparency, report the split IDs (save to `outputs/val_patient_ids_<n>.txt`) and include a sensitivity analysis comparing results across multiple random seeds.

Commands to run (example, 55-val patients):
```
python train.py --data-dir ./data --val-count 55 --max-patients 80 --epochs 40 --batch-size 8 --save-dir models
python scripts/evaluate_3d_unet_monai.py --checkpoint models/best_unet3d_monai.pt --data-dir ./data --max-patients 80 --val-count 55 --baseline-2d-checkpoint models/best_unet.pt --baseline-2d-model unet --output-json outputs/compare_2d_3d_brats_regions_55.json --output-csv outputs/compare_2d_3d_brats_regions_55.csv
python scripts/evaluate_patient_level_wilcoxon.py --data-dir ./data --max-patients 80 --val-count 55 --unet-checkpoint models/best_unet.pt --resunet-checkpoint models/best_resunet.pt --output-dir outputs
```

Caveats
- If you use `--max-patients`, ensure `--val-count < --max-patients` (the scripts require at least one train patient).
- If you rely on an earlier checkpoint that stored `val_patient_ids`, evaluation prefers those IDs — to force a fresh split, either remove `val_patient_ids` from the checkpoint or pass `--val-count` and `--seed` to re-split.

If you want, I can:
- Save the picked 55 validation IDs to `outputs/val_patient_ids_55.txt` now, or
- Run the Wilcoxon evaluation here with `val_count=55` (this will run models and may be time-consuming), or
- Produce a short paragraph for Section 5.5 incorporating this limitation and the proposed remedial analysis.
