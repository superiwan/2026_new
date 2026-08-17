# Poker Corner YOLO Real-Domain Fine-Tune v3

This report records the green-A4 fine-tune requested after the black-background
domain-gap check. It is a diagnostic/model-development result, not a claim of
independent real-scene accuracy.

## Source and Rectification

The four raw frames are `D:\26_merge\piture\fall2\5.jpg` through `8.jpg`.
They were copied read-only and rectified by the existing project
`legacy_2026_new.detect_a4()` followed by `legacy_2026_new.warp_a4()` to
`420x594` pixels. The exact detected camera quads and source paths are in
`D:\26_merge_artifacts\poker_yolo\data_train\real_finetune_v3\green_sources\a4_warp_lineage.json`.

The `fall` directory contains MaixVision UI screenshots. It was deliberately
not used for v3 because overlays contaminate the image. No pseudo-labels were
created from any model output.

## Manual Labels

The following boxes are `xyxy` pixels in the warped `420x594` images. Each image
has exactly two complete corner marks:

```text
5_a4.png: (229,54,296,93), (71,196,148,231)       # JOKER, JOKER
6_a4.png: (217,400,291,439), (98,478,192,516)      # JOKER, JOKER
7_a4.png: (286,27,331,79), (214,203,257,253)       # 10H, 10H
8_a4.png: (103,153,148,191), (270,66,312,111)     # 9S, 9S
```

Yellow-box review images and the source labels are under
`real_finetune_v3\dataset_green_real_ft_v3\manual_audit_green_sources`.

## Dataset

`scripts/build_poker_green_finetune_dataset.py` combines:

- 1,000 synthetic train images, 180 synthetic validation images and 180 synthetic test images;
- 240 green-A4 augmented train images and 60 green-A4 augmented validation images;
- 32 retained black-background augmented train images to reduce catastrophic forgetting.

The augmentation applies bounded global perspective, rotation, scale, exposure,
blur and noise. Horizontal and vertical mirrors are disabled. All green
validation images originate from the same four source frames, so this validation
is source-overlapping. The test split is synthetic-only and is reported as such.
`scripts/validate_yolo_dataset.py` reports no malformed labels or unreadable
images.

## Training and Artifacts

The model started from the black-domain v2 `best.pt`, using official YOLO11n,
`imgsz=640`, batch 16, 15 epochs, `rect=false`, `cache=false`, and the existing
rotation/perspective/lighting policy. The run is:

```powershell
$py = 'D:\26_电赛省赛\.venv_yolo\Scripts\python.exe'
& $py scripts\train_poker_corner_yolo.py `
  --data D:\26_merge_artifacts\poker_yolo\data_train\real_finetune_v3\dataset_green_real_ft_v3 `
  --project D:\26_merge_artifacts\poker_yolo\data_train\real_finetune_v3\runs `
  --name poker_corner_yolo11n_640_green_realft_v3 `
  --model D:\26_merge_artifacts\poker_yolo\data_train\real_finetune_v2\runs\poker_corner_yolo11n_640_realft_v2\weights\best.pt `
  --epochs 15 --batch 16 --workers 0
```

Artifacts are under
`D:\26_merge_artifacts\poker_yolo\data_train\real_finetune_v3\runs\poker_corner_yolo11n_640_green_realft_v3\weights`:

- `best.pt`
- `best.onnx` (fixed `images [1,3,640,640] -> output0 [1,5,8400]`)
- `best_maix_raw.onnx` (fixed `images [1,3,640,640] -> DFL [1,1,4,8400]` and `Sigmoid [1,1,8400]`)

The raw ONNX passed `onnx.checker` and ONNX Runtime metadata inspection. The
derived v2 CV181x INT8 artifact is stored separately under
`D:\26_merge\output\poker_corner_conversion_640_v2` and was loaded on the
MaixCAM Pro as a `640x640` RGB one-class YOLO11 model.

## Metrics and Domain-Gap Evidence

Independent synthetic test (180 images, 210 boxes): Precision `0.9800`, Recall
`0.9343`, mAP50 `0.9599`, mAP50-95 `0.8094`.

Same-source diagnostic at `conf=0.25`, matching IoU `0.30`:

| Model | Green warped frames | Black fixtures |
| --- | ---: | ---: |
| Synthetic v1 | not evaluated | `0/10` matched, 7 predictions |
| Black-domain v2 | `2/8` matched, 5 predictions | `0/10` matched, 7 predictions |
| Green-domain v3 | `8/8` matched, 8 predictions | `6/10` matched, 7 predictions |

The green result demonstrates adaptation to these four source frames, not field
generalization. The black result shows retained but imperfect cross-domain
behavior. Do not use these five or four source parents as an independent accuracy
benchmark.

## Device Fixed-Image Replay

The v2 `.cvimodel` and `.mud` were uploaded to `/root/models`. Replaying the four
rectified green-A4 images (`5_a4.png` through `8_a4.png`) on the device produced
exactly two detections per image, with confidence values in the range `0.8047` to
`0.8594`. This is bounded fixed-image evidence only; no live camera, display, or
UART electrical validation is claimed.
