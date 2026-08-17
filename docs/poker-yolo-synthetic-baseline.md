# Poker Corner YOLO Synthetic Baseline

This experiment is intentionally independent from the production solver. It trains one class,
`corner_mark`, to locate a complete rank-plus-suit card corner. It does not infer rank, suit,
orientation, fragment assignment, or the original/swapped solution.

## Data lineage

`scripts/generate_poker_corner_synthetic.py` creates every image and annotation from primitive
shapes, text and suit drawings. It does not download card photos, so the generated corpus and
labels are released as CC0-1.0. The attempted public source candidate was
`https://github.com/geaxgx/playing-card-detection` (repository code is MIT according to its
repository metadata), but the current network request timed out and its image-asset rights were
not independently verified. It is therefore not part of this dataset.

The generated scenes contain a green A4-like rectangle, white card fragments, full top-left or
bottom-right corner marks, arbitrary in-plane rotation, small perspective perturbation, shadows,
lighting variation, blur, noise and JPEG compression. About 23 percent of scenes are empty-label
hard negatives; additional negative fragments contain central suit graphics without a valid corner
mark. Images and labels are split by disjoint deterministic random seeds.

This is a pipeline baseline only. It has no MaixCAM capture data, real lens distortion, actual
card print texture, true cutting artifacts, or contest lighting. Its test metrics must never be
reported as real-scene or device accuracy.

## Reproduction

```powershell
$py = 'D:\26_电赛省赛\.venv_yolo\Scripts\python.exe'
$artifact = 'D:\26_merge_artifacts\poker_yolo\data_train'
& $py scripts\generate_poker_corner_synthetic.py --output "$artifact\dataset_synth_v2"
& $py scripts\validate_yolo_dataset.py --data "$artifact\dataset_synth_v2\data.yaml"
& $py scripts\train_poker_corner_yolo.py --data "$artifact\dataset_synth_v2" --project "$artifact\runs" --model "$artifact\source_weights\yolo11n.pt" --epochs 80 --batch 8
```

Training uses official `yolo11n.pt`, `imgsz=640`, `patience=20`, `rect=false`, `cache=false`,
rotation/perspective/lighting augmentation and no horizontal or vertical mirroring. The completed
short run used RTX 4060 Laptop GPU batch 16; use batch 8 for the full dataset if other GPU load
makes batch 16 unstable. The model is exported as fixed `1x3x640x640` ONNX with opset 17 and
`dynamic=False`.

The short actual baseline uses `dataset_synth_v3_lite` (240/60/60) and 12 epochs at batch 16 to
fit the available execution window. Its test metrics are pipeline evidence only. The full
`dataset_synth_v2` (1000/180/180) remains validated for a longer 80-epoch run.

After export, inspect actual ONNX input/output node names before any CV181x conversion. This run's
normal ONNX exposes decoded `output0`; the MaixCAM conversion needs two raw output nodes instead.
Create the conversion input with:

```powershell
& $py scripts\extract_maix_yolo11_raw_onnx.py `
  --source "$artifact\runs\poker_corner_yolo11n_640_synth_lite_v1\weights\best.onnx" `
  --output "$artifact\runs\poker_corner_yolo11n_640_synth_lite_v1\weights\best_maix_raw.onnx"
```

## Completed PC Evidence

Run: `poker_corner_yolo11n_640_synth_lite_v1`.

| Item | Result |
| --- | --- |
| Data | 240 train / 60 val / 60 test images; 307 / 85 / 64 `corner_mark` boxes |
| Empty-label negatives | 56 / 9 / 18 images |
| Training | 12 epochs, batch 16, RTX 4060 Laptop GPU, 0.021 hours |
| Test Precision | 0.981858 |
| Test Recall | 0.921875 |
| Test mAP50 | 0.943468 |
| Test mAP50-95 | 0.541322 |
| Standard ONNX | `images [1,3,640,640]` to `output0 [1,5,8400]` |
| Maix raw ONNX | `images [1,3,640,640]` to DFL `[1,1,4,8400]` plus Sigmoid `[1,1,8400]` |

Artifacts are under `D:\26_merge_artifacts\poker_yolo\data_train`. `best.pt`, `best.onnx`,
`best_maix_raw.onnx`, `test_metrics.json`, `args.yaml`, curves and test prediction images are
preserved there. `best_maix_raw.onnx` passed `onnx.checker` and ONNX Runtime metadata inspection;
it has not yet passed CV181x INT8 conversion or device inference.
