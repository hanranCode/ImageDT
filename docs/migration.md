# Python 3 migration: 0.4.1 → 0.6.0rc1

## Compatibility policy

Original public function/class names, module paths and argument names are checked against `tests/fixtures/legacy_api.json`, captured from the original Git revision. Optional implementations may be re-exported from shared modules. Defaults that used mutable lists now use tuples or `None`. The default LeNet prediction callback is resolved on invocation, and produces softmax predictions as before. Accidental exports of third-party module objects (`np`, `os`, `tf`, etc.) are not public compatibility promises. `from imagedt import *` loads all documented exports, including optional plotting; use specific imports if extras are not installed.

Old test filenames in `test/` now launch maintained tests. Dataset-specific demonstration code was replaced with temporary-fixture regression tests. Caffe/Torch trainer packages were empty placeholders; they remain importable placeholders.

## Corrected behavior

| Area | New behavior and reason |
|---|---|
| Directory traversal | Uses `scandir`, sorted names, safe extension extraction, copied filters and no recursion into directory symlinks. Results are reproducible and cyclic links terminate. |
| Image cleanup | Thread workers avoid macOS process-pickling failures. All workers finish before `clear_dir` returns; every file is checked, including leftovers beyond equal-size chunks. |
| Padding | Integer sizes, positive dimension validation, bounded resize dimensions and aspect-preserving rectangular fit. Extremely thin images remain valid. |
| Resize/hash/draw | Requested resize bound is honored; `imwrite` argument order is corrected; binary streaming hashes work on image bytes; polygons become standard OpenCV `(N,4,1,2)` contours. |
| File pairs | Multi-dot stems are preserved, annotation membership checks use a set, pair renames preflight collisions and roll back completed renames on a subsequent failure. MD5 image/XML renames share the same collision preflight and rollback. No filesystem transaction is promised against concurrent writers or process termination. |
| Files | UTF-8 text, BOM-aware CSV with correct newline handling, optional XLS dependency and byte decoding. |
| XML | Binary serialization, disabled external entity resolution, safe defaults, stable empty polygon tuple, consistent list coordinates and list-based mutation. |
| Detection F1 | Uses actual/explicit labels, confidence filtering, class-aware greedy matching and VOC inclusive boxes with strict `IoU > threshold`. Duplicate/unmatched detections count as FP, unmatched non-difficult GT as FN, difficult GT are ignored, empty scores are zero. Class 0 remains excluded. Numeric results may differ because old matching was incorrect. |
| Classification F1 | Aligns input files by basename, rejects duplicate/missing image IDs, keeps the header out of numeric sorting, populates object IDs even with zero true positives, and handles empty totals. |
| ROC | Preserves the 1001 threshold grid and output filenames; sorted probability arrays with `searchsorted` replace the nested Python scan. Rejects invalid probabilities and closes generated figures. This reports confidence-versus-correctness, following the original contract; it is not a general multiclass ROC API. |
| Dataset batches | Preserves shuffle-with-paired-labels, fixed batches discard incomplete tails, non-fixed batches include tails. Empty/mismatched data and impossible fixed batch sizes raise `ValueError` instead of recursing indefinitely. |
| TFRecord | Shared serialization implementation; output directories are created; emitted PNG bytes advertise PNG; no empty extra shard; validation samples are excluded from training; empty detection annotations serialize correctly, corrupt-image reports are split-specific, and converter-generated classification labels use 0..N-1 for one-hot compatibility. Classification regeneration stages new shards before publication and removes obsolete `train_######.tfrecord` / `validation_######.tfrecord` files (exactly six digits), preserving unrelated filenames. Publication across multiple files is not atomic. Existing one-based classification indexes must be remapped before Data_Provider use. |
| Model wrappers | Isolated graphs and explicit session closure; top-class selection uses argmax rather than a complete sort; tied maximum scores select the first class. |
| Caffe graph | Descriptor-pool protobuf, iterative linear-time DAG ordering, updated node lookup after renaming, correct exclude-phase filtering, integer grouped-convolution channels, current split/concat signatures and stable FC/bias dimensions. Serialized BatchNorm statistics are normalized correctly and protobuf Scale gamma is preserved in supported BatchNorm/Scale fusion. Terminal output layers retain a stable graph alias, and unsupported padding/layers fail explicitly. SAME is accepted only when its effective padding equals Caffe symmetric padding; equal output shape alone is insufficient. Generated Python network names are validated and string arguments quoted. |
| LeNet/losses | LeNet uses maintained Keras and retains function/endpoint names. TF-Slim checkpoint variable names change. ArcFace normalizes zero vectors safely and clamps square-root rounding errors. |
| Synthetic text | Pillow metrics/resampling updated; font size, image size and orientation arguments are honored. Reusable thread pools have `close()` and context managers; batches preserve ordering. Quasicrystal backgrounds are vectorized; distortion arrays use uint8. Outputs will not reproduce the old random font-size/half-size digit layout. |
| PDF | PDFium replaces native ImageMagick/Ghostscript requirements. Multiple pages have deterministic indexed filenames. |

## Explicit unsupported operations

`caffe_coreml.convert`, `HandwrittenTextGenerator.generate`, `FakeTextDataGenerator.generate(is_handwritten=True)`, old TOCO DOT output, and old uncalibrated uint8 conversion raise `NotImplementedError`. The user selected preservation of old entrances with a modern backend and explicit failure when compatibility cannot be maintained. No emulation of deleted upstream facilities is claimed.

Model integration tests exercise small synthetic models and the supported layer subset. They do not establish equivalence for every historical Caffe model, custom layer or trained TF-Slim checkpoint. For those workloads, add a representative fixture and compare outputs before switching environments.

## Backend references

- [TensorFlow compatibility converter](https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter)
- [Google LiteRT migration](https://developers.google.com/edge/litert/migration)
- [Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
