# ImageDT

Python 3 utilities for image processing, files, annotations, metrics, synthetic text, TFRecords and model conversion. The original module paths, function names and public argument names are retained, including historical spellings such as `noise_padd`, `fillter_file_type`, `wirte_xml_file` and `gernerate_infos_dict`.

## Install

Requires Python 3.10 or newer. The candidate release is **0.6.0rc1**.

```bash
python -m pip install imagedt
# Optional capabilities:
python -m pip install 'imagedt[plot,excel,pdf]'
python -m pip install 'imagedt[tensorflow]'  # TensorFlow 2 + LiteRT
python -m pip install 'imagedt[lite]'        # LiteRT inference without TensorFlow
python -m pip install 'imagedt[caffe]'       # Caffe protobuf/graph tooling
```

For this checkout, use `python -m pip install -e '.[dev,plot,excel,pdf,tensorflow]'`. The candidate is only available on PyPI after publication. OpenCV's headless package supports processing without requiring a desktop display; `cv_show` needs a GUI-enabled OpenCV installation. Install exactly one OpenCV distribution in an environment.

Importing `imagedt` does not import TensorFlow, matplotlib, OpenCV or spreadsheet backends. Capabilities load when their exported names are accessed. An unavailable optional backend reports the extra to install.

## Familiar interfaces

```python
import imagedt

paths = imagedt.loop('./images', ['.jpg', '.png'])
imagedt.write_csv([['name', 'count'], ['图片', 3]], 'counts.csv')
boxes = imagedt.Anno_OP.parse_lab_xml('annotations/example.xml')
```

`FilesOp`, `Anno_OP`, `RecordWriter` and `plot_tools` remain singleton objects at their historical import paths. Traversal is deterministic. Text files use UTF-8; CSV readers also accept a UTF-8 BOM.

```python
from imagedt.tensorflow.tools.tfmodel_wrapper import TFmodel_Wrapper

with TFmodel_Wrapper('model.pb', input_nodename='data',
                     output_nodename='softmax,features') as model:
    classifications = model.predict(images)
    features = model.extract(images)
```

Graph wrappers own isolated sessions and support `close()` and context managers. Importing a backend does not disable eager execution or change GPU environment variables. TF1 graph APIs use `tf.compat.v1`; queue-based `Data_Provider` should be created within `tf.Graph().as_default()` and run with a Session and queue-runner lifecycle.

## Behavioral corrections and migration limits

See [the migration guide](docs/migration.md) for precise changes. Important corrections include class-aware detection matching, filename-aligned classification evaluation, no train/validation leakage, bounded dataset batching, binary XML writing, complete image cleanup, and collision-aware paired renaming.

Maintained conversion support covers the bundled Caffe layer subset to frozen GraphDef, frozen GraphDef to float TFLite, and float32 single-image NHWC LiteRT classification. Arbitrary Caffe variants or model checkpoints are not guaranteed. LeNet uses Keras layers while retaining endpoint keys; **old TF-Slim variable/checkpoint names are not preserved**.

These old capabilities retain callable entrances and raise explicit errors:

- Caffe-to-CoreML: removed from modern coremltools; first migrate the model to a supported format.
- Handwriting generation: the external TF1 pretrained model was never bundled. Computer text generation remains supported.
- Legacy TOCO `GRAPHVIZ_DOT` and uint8 quantization without representative calibration data.

PDF rendering uses portable PDFium wheels rather than ImageMagick/Ghostscript. Single-page output keeps the requested filename; multiple pages use `<stem>-0`, `<stem>-1`, etc.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for Python conventions, branches and contribution workflow; [TESTING.md](TESTING.md) for checks and test lanes; [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Develop and verify

```bash
python -m pip install -e '.[dev,plot,excel,pdf,tensorflow]'
python -m ruff check imagedt tests scripts test
python -m ruff format --check imagedt tests scripts test setup.py
python -m pytest --cov=imagedt --cov-report=term-missing
python -m build
python -m twine check dist/*
python scripts/benchmark_utils.py
```

Tests generate their own images, XML, fonts and tiny models in temporary directories. No personal `/data` paths, production datasets or GPU are needed. TensorFlow tests skip explicitly when the optional backend is absent. `tests/fixtures/legacy_api.json` captures public names and argument names from the original revision and guards compatibility. It does not claim numerical equivalence for intentionally corrected bugs or unchanged trained-checkpoint compatibility.

CI defines Python 3.10/3.12/3.14 core jobs on Linux, macOS and Windows, plus a Python 3.12 TensorFlow/LiteRT integration job. Remote CI execution is separate from local verification.

## Release

`bash upload_pip.sh` builds into a fresh temporary directory and checks metadata. It does not upload. Publish only after reviewing the artifacts and configuring PyPI credentials:

```bash
PYTHON=python3 bash upload_pip.sh --upload
```

See [release validation](docs/release-readiness.md) and [changes](CHANGELOG.md). License: Apache-2.0.

The [historical technical review](docs/technical-reviews/2026-10-12-78c9e67/technical-review.md) records the original revision baseline, architecture and migration decisions.
