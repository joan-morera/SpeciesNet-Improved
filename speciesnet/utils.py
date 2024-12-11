# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities."""

__all__ = [
    "only_one_true",
    "file_exists",
    "load_rgb_image",
    "prepare_instances_dict",
]

from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Optional, Union

from absl import logging
from cloudpathlib import CloudPath
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import requests

StrPath = Union[str, Path]

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported extensions for image discovery.
IMG_EXTENSIONS = {
    "jpg",
    "JPG",
    "jpeg",
    "JPEG",
    "png",
    "PNG",
    "tif",
    "TIF",
    "tiff",
    "TIFF",
    "webp",
    "WEBP",
}

# Custom agent for image requests over HTTP(S).
CUSTOM_HTTP_AGENT = {"User-Agent": "SpeciesNetBot/0.0 (FIXME_Homepage)"}


def only_one_true(*args) -> bool:
    """Checks that only one of the given arguments is `True`."""

    already_found = False
    for arg in args:
        if arg:
            if already_found:
                return False
            else:
                already_found = True
    return already_found


def file_exists(filepath_or_url: str) -> bool:
    """Checks whether a given file exists and is accessible.

    Args:
        filepath_or_url:
            String representing either a local file, an `http(s)://` URL or a cloud
            location (identified by one of these prefixes: `az://`, `gs://`, `s3://`).

    Returns:
        `True` if file exists and is accessible, or `False` otherwise.
    """

    try:
        if filepath_or_url.startswith("http://") or filepath_or_url.startswith(
            "https://"
        ):
            return requests.get(
                filepath_or_url, headers=CUSTOM_HTTP_AGENT, timeout=60
            ).ok
        elif (
            filepath_or_url.startswith("az://")
            or filepath_or_url.startswith("gs://")
            or filepath_or_url.startswith("s3://")
        ):
            return CloudPath(filepath_or_url).exists()  # type: ignore
        else:
            return Path(filepath_or_url).exists()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "`%s` while loading `%s` ==> %s", type(e).__name__, filepath_or_url, e
        )
        return False


def load_rgb_image(filepath_or_url: str) -> Optional[PIL.Image.Image]:
    """Loads a file as an RGB PIL image.

    Args:
        filepath_or_url:
            String representing either a local file, an `http(s)://` URL or a cloud
            location (identified by one of these prefixes: `az://`, `gs://`, `s3://`).

    Returns:
        An RGB PIL image if the file was loaded successfully, or `None` otherwise.
    """

    try:
        if filepath_or_url.startswith("http://") or filepath_or_url.startswith(
            "https://"
        ):
            file_contents = requests.get(
                filepath_or_url, headers=CUSTOM_HTTP_AGENT, timeout=60
            ).content
            img = PIL.Image.open(BytesIO(file_contents))
        else:
            if (
                filepath_or_url.startswith("az://")
                or filepath_or_url.startswith("gs://")
                or filepath_or_url.startswith("s3://")
            ):
                path = CloudPath(filepath_or_url)  # type: ignore
            else:
                path = Path(filepath_or_url)
            img = PIL.Image.open(path)

        img.load()
        img = img.convert("RGB")
        img = PIL.ImageOps.exif_transpose(img)
        return img

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "`%s` while loading `%s` ==> %s", type(e).__name__, filepath_or_url, e
        )
        return None


def prepare_instances_dict(
    instances_dict: Optional[dict] = None,
    instances_json: Optional[StrPath] = None,
    filepaths: Optional[list[StrPath]] = None,
    filepaths_txt: Optional[StrPath] = None,
    folders: Optional[list[StrPath]] = None,
    folders_txt: Optional[StrPath] = None,
) -> dict:
    """Transforms various input formats into an instances dict.

    The instances dict is the most expressive input format of them all since, compared
    to others, it can also express country, latitude and longitude information.

    This function expects that only one input argument is provided.

    Args:
        instances_dict:
            Optional instances dict. If provided, this function is a no-op.
        instances_json:
            Optional path to load the instances dict from.
        filepaths:
            Optional list of filepaths to process.
        filepaths_txt:
            Optional path to load the list of filepaths to process from.
        folders:
            Optional list of folders to process.
        folders_txt:
            Optional path to load the list of folders to process from.

    Returns:
        An instances dict resulted from the input transformation.

    Raises:
        ValueError:
            If more than one input argument was provided.
    """

    inputs_str = (
        "["
        "instances_dict, "
        "instances_json, "
        "filepaths, "
        "filepaths_txt, "
        "folders, "
        "folders_txt"
        "]"
    )
    inputs = eval(inputs_str)
    if not only_one_true(*inputs):
        raise ValueError(
            f"Expected exactly one of {inputs_str} to be provided. "
            f"Received: {inputs}."
        )

    if instances_json is not None:
        with open(instances_json, mode="r", encoding="utf-8") as fp:
            instances_dict = json.load(fp)
    if instances_dict is not None:
        return instances_dict

    if folders_txt is not None:
        with open(folders_txt, mode="r", encoding="utf-8") as fp:
            folders = [line.strip() for line in fp.readlines()]
    if folders is not None:
        filepaths = []
        for folder in folders:
            base_dir = Path(folder)
            for ext in IMG_EXTENSIONS:
                filepaths.extend(base_dir.glob(f"**/*.{ext}"))
        filepaths = sorted(filepaths)

    if filepaths_txt is not None:
        with open(filepaths_txt, mode="r", encoding="utf-8") as fp:
            filepaths = [line.strip() for line in fp.readlines()]
    assert filepaths is not None
    return {"instances": [{"filepath": str(filepath)} for filepath in filepaths]}


def load_partial_predictions(
    predictions_json: Optional[StrPath], instances: list[dict]
) -> tuple[dict[str, dict], list[dict]]:
    """Loads partial predictions and filters unprocessed instances from a given list.

    Args:
        predictions_json:
            Filepath to partial predictions to load. If missing, no previous predictions
            can be reused and all instances are considered unprocessed.
        instances:
            List of instances to check if they still need to be processed. Those found
            in the partial predictions are considered already processed and are filtered
            out.

    Returns:
        A tuple made of: (a) the partial predictions dict, and (b) the list of
        unprocessed instances.

    Raises:
        RuntimeError:
            If the partial predictions contain a filepath not found in the list of
            instances to check. To fix this, make sure that the partial predictions
            originated from the same list of instances to process.
    """

    if not predictions_json:
        return {}, instances
    predictions_json = Path(predictions_json)
    if not predictions_json.exists() or not predictions_json.is_file():
        return {}, instances

    logging.info("Loading partial predictions from `%s`.", predictions_json)

    partial_predictions = {}
    target_filepaths = {instance["filepath"] for instance in instances}
    with open(predictions_json, mode="r", encoding="utf-8") as fp:
        predictions_dict = json.load(fp)
        for prediction in predictions_dict["predictions"]:
            if prediction["filepath"] not in target_filepaths:
                raise RuntimeError(
                    f"Filepath from loaded predictions is missing from the set of "
                    f"instances to process: `{prediction['filepath']}`. Make sure "
                    f"you're resuming the work using the same set of instances."
                )
            if "failures" in prediction:
                continue
            partial_predictions[prediction["filepath"]] = prediction

    instances_to_process = [
        instance
        for instance in instances
        if instance["filepath"] not in partial_predictions
    ]

    return partial_predictions, instances_to_process


def save_predictions(predictions_dict: dict, output_json: StrPath) -> None:
    """Saves a predictions dict to an output file.

    Args:
        predictions_dict:
            Predictions dict to save.
        output_json:
            Output filepath where to save the predictions dict in JSON format.
    """

    output_json = Path(output_json)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_json.parent,
        prefix=f"{output_json.name}.tmp.",
        delete=False,
    ) as fp:
        logging.info("Saving predictions to `%s`.", fp.name)
        output_json_tmp = Path(fp.name)
        json.dump(predictions_dict, fp, ensure_ascii=False, indent=4)
    logging.info("Moving `%s` to `%s`.", output_json_tmp, output_json)
    output_json_tmp.replace(output_json)  # Atomic operation.
