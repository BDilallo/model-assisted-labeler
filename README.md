# Model-Assisted Labeler

A desktop tool for building YOLO-format object detection datasets, with an optional YOLO model in the loop to speed up labeling. Built with PySide6.

## How it works

Everything is organized around **sessions**. A session points at a folder of source images (read-only, never modified or written into) and optionally a YOLO model (`.pt`, `.onnx`, `.engine`) used to pre-populate bounding boxes. When you save an image, the app copies it and writes a YOLO-format label file into a session-owned folder inside your chosen workspace, leaving the original files untouched. That saved set is the "annotation pool" for the session, and it's what gets exported later.

Sessions are saved and can be reopened any time; a review step on load lets you fix up the image/model paths if they've moved.

## Labeling

- Draw, resize, move, and delete bounding boxes directly on the canvas, with zoom and fit-to-image.
- Manage a per-session list of classes, assign a box to a class, and add or remove classes as needed (with checks against classes still in use).
- Run the loaded model on the current image ("Predict / Refresh") to add predicted boxes, or replace all current boxes with a fresh prediction.
- Auto Predict mode runs prediction automatically when you land on an image that has no existing or preserved annotations, so you're not repeatedly hitting the same button.
- Batch Auto Annotate runs the model over every clean, unsaved image at once (with configurable confidence threshold, batch size, and CPU/GPU/auto device selection), saving only the images that end up with qualifying boxes.
- Boxes are tracked by where they came from — manual, straight from the model, or a model box you've since edited — which the filters and canvas colors both reflect.

## Filtering and navigation

The filter bar narrows down which images Back/Next/Save & Next will walk through, without touching image order or deleting anything. You can filter by saved/unsaved state, presence of boxes, confidence thresholds, annotation source (manual/model/edited), box count, or by a specific class (including images missing a class entirely).

## Export

Once you've saved a batch of images, Export Dataset builds a ready-to-train YOLO folder (`images/`, `labels/`, `classes.txt`, `data.yaml`) with a train/val split by percentage or count, optional shuffling with a repeatable seed, and optional class ID remapping to close gaps left by deleted classes.

## Other conveniences

- Progress dialogs (with cancel) for anything that scans or processes a lot of images: opening a session, batch annotation, and export.
- Nearby images are pre-fetched in the background while you work so navigation stays snappy.
- Image dimensions are cached to disk so re-opening a large session doesn't need to re-read every file.
- Unsaved-change tracking with save prompts on exit, plus a "Clear All Images" reset that requires typing a confirmation phrase.

## Running it

Requires Python with PySide6 installed, and `ultralytics` (plus a CUDA-enabled `torch` if you want GPU inference) for model-assisted prediction — the app still runs without a model, just without predictions.

```
python -m model_assisted_labeler.app
```

---

This is very much a work in progress — expect rough edges and things to shift around as it keeps getting built out.
