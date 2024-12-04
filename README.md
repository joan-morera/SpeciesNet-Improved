# SpeciesNet

## Overview

This repository hosts code for running an ensemble of two models: (1) an object detector that finds objects of interest in wildlife camera images, and (2) an image classifier that classifies those objects to the species level. This ensemble is used for species recognition in the [Wildlife Insights](https://www.wildlifeinsights.org/) platform.

The species classifier was trained at Google, while the object detector is the publicly-available [MegaDetector](https://github.com/agentmorris/MegaDetector).

## Getting started

First, to use the code and model weights from this repository, you need to clone it:

```bash
git clone https://github.com/google/FIXME.git

cd FIXME
```

## Running the code

We recommend that you create a Python virtual environment as follows:

```bash
python -m venv .env

source .env/bin/activate
```

Depending on how you plan to run the SpeciesNet, you can install either:

- minimal requirements

    ```bash
    pip install -e .
    ```

- minimal + notebooks requirements

    ```bash
    pip install -e .[notebooks]
    ```

- minimal + server requirements

    ```bash
    pip install -e .[server]
    ```

- minimal + cloud requirements (`az` / `gs` / `s3`), e.g.

    ```bash
    pip install -e .[gs]
    ```

- any combination of the above requirements, e.g.

    ```bash
    pip install -e .[notebooks,server]
    ```

Once you have installed the necessary dependencies, you have several ways of running SpeciesNet:

1. Via a restartable script for one-off jobs by setting one of the following flags: `--instances_json`, `--filepaths`, `--filepaths_txt`, `--folders`, `--folders_txt`.

    ```bash
    python scripts/run_model.py \
        --instances_json=test_data/instances.json \
        --predictions_json=predictions.json
    ```

1. Via a local prediction server you can launch with:

    ```bash
    python scripts/run_server.py
    ```

   and query with:

    ```bash
    curl \
        -H "Content-Type: application/json" http://0.0.0.0:8000/predict \
        -d "@test_data/instances.json"
    ```

1. Programmatically via the internal API. See [this notebook](notebooks/run_model.ipynb) for several examples.

## Supported models

- [v4.0.0a](model_cards/v4.0.0a) (default): Always crop model, i.e. we run the detector first and crop the image to the top detection bounding box before feeding it to the species classifier.
- [v4.0.0b](model_cards/v4.0.0b): Full image model, i.e. we run both the detector and the species classifier on the full image, independently.

## Input schema

SpeciesNet runs inference on instances dicts in the following format. When you call the model, you can either prepare your requests to match this format or, in some cases, other supported formats will be converted to this automatically.

```text
{
    "instances": [
        {
            "filepath": str  => Image filepath.
            "country": str (optional)  => 3-letter country code (ISO 3166-1 Alpha-3) for the location where the image was taken.
            "latitude": float (optional)  => Latitude where the image was taken.
            "longitude": float (optional)  => Longitude where the image was taken.
        },
        ...  => A request can contain multiple instances in the format above.
    ]
}
```

## Output schemas

When you receive a response from SpeciesNet, it will be in one of the following formats based on what functionality you requested.

### Full inference

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failures": list[str] (optional)  => List of internal components that failed during prediction (e.g. "CLASSIFIER", "DETECTOR", "GEOLOCATION"). If absent, the prediction was successful.
            "country": str (optional)  => 3-letter country code (ISO 3166-1 Alpha-3) for the location where the image was taken. It can be overwritten if the country from the request doesn't match the country of (latitude, longitude).
            "admin1_region": str (optional)  => First-level administrative division for the (latitude, longitude) in the request. Included only for some countries that are used in geofencing (e.g. "USA").
            "latitude": float (optional)  => Latitude where the image was taken, included only if (latitude, longitude) were present in the request.
            "longitude": float (optional)  => Longitude where the image was taken, included only if (latitude, longitude) were present in the request.
            "classifications": {  => dict (optional)  => Top-5 classifications. Included only if "CLASSIFIER" if not part of the "failures" field.
                "classes": list[str]  => List of top-5 classes predicted by the classifier, matching the decreasing order of their scores below.
                "scores": list[float]  => List of scores corresponding to top-5 classes predicted by the classifier, in decreasing order.
            },
            "detections": [  => list (optional)  => List of detections with confidence scores > 0.01, in decreasing order of their scores. Included only if "DETECTOR" if not part of the "failures" field.
                {
                    "category": str  => Detection class "1" (= animal), "2" (= human) or "3" (= vehicle) from MegaDetector's raw output.
                    "label": str  => Detection class "animal", "human" or "vehicle", matching the "category" field above. Added for readability purposes.
                    "conf": float  => Confidence score of the current detection.
                    "bbox": list[float]  => Bounding box coordinates, in (xmin, ymin, width, height) format, of the current detection. Coordinates are normalized to the [0.0, 1.0] range, relative to the image dimensions.
                },
                ...  => A prediction can contain zero or multiple detections.
            ],
            "prediction": str (optional)  => Final prediction of the SpeciesNet ensemble. Included only if "CLASSIFIER" and "DETECTOR" are not part of the "failures" field.
            "prediction_score": float (optional)  => Final prediction score of the SpeciesNet ensemble. Included only if the "prediction" field above is included.
            "prediction_source": str (optional)  => Internal component that produced the final prediction. Used to collect information about which parts of the SpeciesNet ensemble fired. Included only if the "prediction" field above is included.
            "exif": {  => dict (optional)  => Relevant EXIF fields extracted from the image metadata.
                "DateTimeOriginal": str (optional)  => Date and time when the original image was captured.
            }
            "model_version": str  => A string representing the version of the model that produced the current prediction.
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

### Classifier-only inference

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failure": str (optional)  => Failure message encountered during prediction. If absent, the prediction was successful.
            "classifications": {  => dict (optional)  => Top-5 classifications. Included only if "CLASSIFIER" if not part of the "failures" field.
                "classes": list[str]  => List of top-5 classes predicted by the classifier, matching the decreasing order of their scores below.
                "scores": list[float]  => List of scores corresponding to top-5 classes predicted by the classifier, in decreasing order.
            }
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

### Detector-only inference

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failure": str (optional)  => Failure message encountered during prediction. If absent, the prediction was successful.
            "detections": [  => list (optional)  => List of detections with confidence scores > 0.01, in decreasing order of their scores. Included only if "DETECTOR" if not part of the "failures" field.
                {
                    "category": str  => Detection class "1" (= animal), "2" (= human) or "3" (= vehicle) from MegaDetector's raw output.
                    "label": str  => Detection class "animal", "human" or "vehicle", matching the "category" field above. Added for readability purposes.
                    "conf": float  => Confidence score of the current detection.
                    "bbox": list[float]  => Bounding box coordinates, in (xmin, ymin, width, height) format, of the current detection. Coordinates are normalized to the [0.0, 1.0] range, relative to the image dimensions.
                },
                ...  => A prediction can contain zero or multiple detections.
            ]
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

## Developing and contributing code

If you're interested in developing code on top of our repo (and hopefully contributing it back!), create the Python virtual environment for development using the following commands:

```bash
python -m venv .env

source .env/bin/activate

pip install -e .[dev]
```

We use:

- [`black`](https://github.com/psf/black) for code formatting:

    ```bash
    black .
    ```

- [`isort`](https://github.com/PyCQA/isort) for sorting Python imports consistently:

    ```bash
    isort .
    ```

- [`pylint`](https://github.com/pylint-dev/pylint) for linting code and flag various issues:

    ```bash
    pylint . --recursive=yes
    ```

- [`pyright`](https://github.com/microsoft/pyright) for static type checking:

    ```bash
    pyright
    ```

- [`pytest`](https://github.com/pytest-dev/pytest/) for testing our code:

    ```bash
    pytest -vv
    ```

If you submit a PR to contribute your code back to this repo, you will be asked to sign a contributor license agreement; see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
