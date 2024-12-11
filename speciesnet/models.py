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

"""Model components of SpeciesNet."""

__all__ = [
    "Classification",
    "Detection",
    "BBox",
    "SpeciesNetClassifier",
    "SpeciesNetDetector",
    "SpeciesNetEnsemble",
]

from dataclasses import dataclass
import enum
import json
from pathlib import Path
import time
from typing import Any, Optional

from absl import logging
from huggingface_hub import snapshot_download
from humanfriendly import format_timespan
import kagglehub
import numpy as np
import PIL.ExifTags
import PIL.Image
import tensorflow as tf
import torch
import torch.backends
import torch.backends.mps
from yolov5.utils.augmentations import letterbox as yolov5_letterbox
from yolov5.utils.general import non_max_suppression as yolov5_non_max_suppression
from yolov5.utils.general import scale_boxes as yolov5_scale_boxes
from yolov5.utils.general import xyxy2xywhn as yolov5_xyxy2xywhn

# Handy type aliases.
PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


class Classification(str, enum.Enum):
    """Enum of common classification values.

    The classifier is not limited to these and can predict other string values as well.
    This enum only contains values with a special meaning during the inference process.
    """

    # pylint: disable=line-too-long
    BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
    ANIMAL = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
    HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
    VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
    UNKNOWN = "f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result"


class Detection(str, enum.Enum):
    """Enum of all possible detection values."""

    ANIMAL = "animal"
    HUMAN = "human"
    VEHICLE = "vehicle"

    @classmethod
    def from_category(cls, category: str) -> Optional["Detection"]:
        """Transforms a numeric category from the detector into an enum value.

        Args:
            category: Numeric category from the detector, provided as a string (e.g.
            "1", "2", "3").

        Returns:
            Enum detection value corresponding to the given numeric category. If
            category is not one of "1", "2" or "3", returns `None`.
        """

        category_to_label = {
            "1": Detection.ANIMAL,
            "2": Detection.HUMAN,
            "3": Detection.VEHICLE,
        }
        return category_to_label.get(category)


class Failure(enum.Flag):
    """Enum of flags used to indicate which model components failed during inference."""

    CLASSIFIER = enum.auto()
    DETECTOR = enum.auto()
    GEOLOCATION = enum.auto()


@dataclass(frozen=True)
class ModelInfo:
    """Dataclass describing SpeciesNet model and its underlying resources to load."""

    version: str  # Model version.
    type_: str  # Model type.
    classifier: Path  # Path to classifier model.
    classifier_labels: Path  # Path to labels file used by clasifier.
    detector: Path  # Path to detector model.
    taxonomy: Path  # Path to taxonomy file used by ensemble.
    geofence: Path  # Path to geofence file used by ensemble.

    def __init__(self, model_name: str) -> None:
        """Creates dataclass to describe a given model.

        Args:
            model_name:
                String value identifying the model to be described by this dataclass.
                It can be a Kaggle identifier (starting with `kaggle:`), a HuggingFace
                identifier (starting with `hf:`) or a local folder to load the model
                from. If the model name is a remote identifier (Kaggle or HuggingFace),
                the model files are automatically downloaded on the first call.
        """

        # Download model files (if necessary) and set the base local directory.
        kaggle_prefix = "kaggle:"
        hf_prefix = "hf:"
        if model_name.startswith(kaggle_prefix):
            base_dir = kagglehub.model_download(model_name[len(kaggle_prefix) :])
        elif model_name.startswith(hf_prefix):
            base_dir = snapshot_download(model_name[len(hf_prefix) :])
        else:
            base_dir = model_name
        base_dir = Path(base_dir)

        # Set dataclass fields using a workaround to bypass read-only constraints.
        with open(base_dir / "info.json", mode="r", encoding="utf-8") as fp:
            info = json.load(fp)
            object.__setattr__(self, "version", info["version"])
            object.__setattr__(self, "type_", info["type"])
            object.__setattr__(self, "classifier", base_dir / info["classifier"])
            object.__setattr__(
                self, "classifier_labels", base_dir / info["classifier_labels"]
            )
            object.__setattr__(self, "detector", base_dir / info["detector"])
            object.__setattr__(self, "taxonomy", base_dir / info["taxonomy"])
            object.__setattr__(self, "geofence", base_dir / info["geofence"])


@dataclass(frozen=True)
class PreprocessedImage:
    """Dataclass describing a preprocessed image."""

    arr: np.ndarray  # Multidimensional array of image pixels.
    orig_width: int  # Original image width.
    orig_height: int  # Original image height.


@dataclass(frozen=True)
class BBox:
    """Dataclass describing a bounding box."""

    xmin: float
    ymin: float
    width: float
    height: float


class SpeciesNetClassifier:
    """Classifier component of SpeciesNet."""

    IMG_SIZE = 480
    MAX_CROP_RATIO = 0.3
    MAX_CROP_SIZE = 400

    def __init__(self, model_name: str) -> None:
        """Loads the classifier resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
        """

        start_time = time.time()

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model_info = ModelInfo(model_name)
        self.model = tf.keras.models.load_model(
            self.model_info.classifier, compile=False
        )
        with open(self.model_info.classifier_labels, mode="r", encoding="utf-8") as fp:
            self.labels = {idx: line.strip() for idx, line in enumerate(fp.readlines())}

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetClassifier in %s.",
            format_timespan(end_time - start_time),
        )

    def preprocess(
        self,
        img: Optional[PIL.Image.Image],
        bboxes: Optional[list[BBox]] = None,
        resize: bool = True,
    ) -> Optional[PreprocessedImage]:
        """Preprocesses an image according to this classifier's needs.

        Args:
            img:
                PIL image to preprocess. If `None`, no preprocessing is performed.
            bboxes:
                Optional list of bounding boxes. Needed for some types of classifiers to
                crop the image to specific bounding boxes during preprocessing.
            resize:
                Whether to resize the image to some expected dimensions.

        Returns:
            A preprocessed image, or `None` if no PIL image was provided initially.
        """

        if img is None:
            return None

        with tf.device("/cpu"):
            img_tensor = tf.convert_to_tensor(img)
            img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

            if self.model_info.type_ == "always_crop":
                # Crop to top bbox if available, otherwise leave image uncropped.
                if bboxes:
                    img_tensor = tf.image.crop_to_bounding_box(
                        img_tensor,
                        int(bboxes[0].ymin * img.height),
                        int(bboxes[0].xmin * img.width),
                        int(bboxes[0].height * img.height),
                        int(bboxes[0].width * img.width),
                    )
            elif self.model_info.type_ == "full_image":
                # Crop top and bottom of image.
                target_height = tf.cast(
                    tf.math.floor(
                        tf.multiply(
                            tf.cast(tf.shape(img_tensor)[0], tf.float32),
                            tf.constant(1.0 - SpeciesNetClassifier.MAX_CROP_RATIO),
                        )
                    ),
                    tf.int32,
                )
                target_height = tf.math.maximum(
                    target_height,
                    tf.shape(img_tensor)[0] - SpeciesNetClassifier.MAX_CROP_SIZE,
                )
                img_tensor = tf.image.resize_with_crop_or_pad(
                    img_tensor, target_height, tf.shape(img_tensor)[1]
                )

            if resize:
                img_tensor = tf.image.resize(
                    img_tensor,
                    [SpeciesNetClassifier.IMG_SIZE, SpeciesNetClassifier.IMG_SIZE],
                )

            img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)
            return PreprocessedImage(img_tensor.numpy(), img.width, img.height)

    def predict(
        self, filepath: str, img: Optional[PreprocessedImage]
    ) -> dict[str, Any]:
        """Runs inference on a given preprocessed image.

        Args:
            filepath:
                Location of image to run inference on. Used for reporting purposes only,
                and not for loading the image.
            img:
                Preprocessed image to run inference on. If `None`, a failure message is
                reported back.

        Returns:
            A dict containing either the top-5 classifications for the given image (in
            decreasing order of confidence scores), or a failure message if no
            preprocessed image was provided.
        """

        if img is None:
            return {
                "filepath": filepath,
                "failure": "Unavailable preprocessed image.",
            }

        img_tensor = tf.convert_to_tensor([img.arr / 255])
        logits = self.model(img_tensor, training=False)
        scores = tf.keras.activations.softmax(logits)
        scores, indices = tf.math.top_k(scores, k=5)
        return {
            "filepath": filepath,
            "classifications": {
                "classes": [self.labels[idx] for idx in indices.numpy()[0]],
                "scores": scores.numpy()[0].tolist(),
            },
        }


class SpeciesNetDetector:
    """Detector component of SpeciesNet."""

    IMG_SIZE = 1280
    STRIDE = 64
    DETECTION_THRESHOLD = 0.01

    def __init__(self, model_name: str) -> None:
        """Loads the detector resources.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
        """

        start_time = time.time()

        self.model_info = ModelInfo(model_name)

        # Select the best device available.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load the model.
        if self.device != "mps":
            checkpoint = torch.load(self.model_info.detector, map_location=self.device)
            self.model = checkpoint["model"].float()
        else:
            checkpoint = torch.load(self.model_info.detector)
            self.model = checkpoint["model"].float().to(self.device)
        self.model.eval()

        # Fix compatibility issues to be able to load older YOLOv5 models with newer
        # versions of PyTorch.
        for m in self.model.modules():
            if isinstance(m, torch.nn.Upsample) and not hasattr(
                m, "recompute_scale_factor"
            ):
                m.recompute_scale_factor = None

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetDetector in %s on %s.",
            format_timespan(end_time - start_time),
            self.device.upper(),
        )

    def preprocess(self, img: Optional[PIL.Image.Image]) -> Optional[PreprocessedImage]:
        """Preprocesses an image according to this detector's needs.

        Args:
            img:
                PIL image to preprocess. If `None`, no preprocessing is performed.

        Returns:
            A preprocessed image, or `None` if no PIL image was provided initially.
        """

        if img is None:
            return None

        img_arr = yolov5_letterbox(
            np.asarray(img),
            new_shape=SpeciesNetDetector.IMG_SIZE,
            stride=SpeciesNetDetector.STRIDE,
            auto=True,
        )[0]
        return PreprocessedImage(img_arr, img.width, img.height)

    def _convert_yolo_xywhn_to_md_xywhn(self, yolo_xywhn: list[float]) -> list[float]:
        """Converts bbox XYWHN coordinates from YOLO's to MegaDetector's format.

        Args:
            yolo_xywhn:
                List of bbox coordinates in YOLO format, i.e.
                [x_center, y_center, width, height].

        Returns:
            List of bbox coordinates in MegaDetector format, i.e.
            [x_min, y_min, width, height].
        """

        x_center, y_center, width, height = yolo_xywhn
        x_min = x_center - width / 2.0
        y_min = y_center - height / 2.0
        return [x_min, y_min, width, height]

    def predict(
        self, filepath: str, img: Optional[PreprocessedImage]
    ) -> dict[str, Any]:
        """Runs inference on a given preprocessed image.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        Args:
            filepath:
                Location of image to run inference on. Used for reporting purposes only,
                and not for loading the image.
            img:
                Preprocessed image to run inference on. If `None`, a failure message is
                reported back.

        Returns:
            A dict containing either the detections above a fixed confidence threshold
            for the given image (in decreasing order of confidence scores), or a failure
            message if no preprocessed image was provided.
        """

        if img is None:
            return {
                "filepath": filepath,
                "failure": "Unavailable preprocessed image.",
            }

        # Prepare model input.
        img_arr = img.arr.transpose((2, 0, 1))  # HWC to CHW.
        img_arr = np.ascontiguousarray(img_arr)
        img_tensor = torch.from_numpy(img_arr).to(self.device)
        img_tensor = img_tensor.float() / 255
        img_tensor = torch.unsqueeze(img_tensor, 0)  # Add batch dimension.

        # Run inference.
        results = self.model(img_tensor, augment=False)[0]
        if self.device == "mps":
            results = results.cpu()
        results = yolov5_non_max_suppression(
            prediction=results,
            conf_thres=SpeciesNetDetector.DETECTION_THRESHOLD,
        )
        results = results[0]  # Drop batch dimension.

        # Process detections.
        detections = []
        results[:, :4] = yolov5_scale_boxes(
            img_tensor.shape[2:],
            results[:, :4],
            (img.orig_height, img.orig_width),
        ).round()
        for result in results:  # (x_min, y_min, x_max, y_max, conf, category)
            xyxy = result[:4]
            xywhn = yolov5_xyxy2xywhn(xyxy, w=img.orig_width, h=img.orig_height)
            bbox = self._convert_yolo_xywhn_to_md_xywhn(xywhn.tolist())

            conf = result[4].item()

            category = str(int(result[5].item()) + 1)
            label = Detection.from_category(category)
            if label is None:
                logging.error("Invalid detection class: %s", category)
                continue

            detections.append(
                {
                    "category": category,
                    "label": label.value,
                    "conf": conf,
                    "bbox": bbox,
                }
            )

        # Sort detections by confidence score.
        detections = sorted(detections, key=lambda det: det["conf"], reverse=True)

        return {
            "filepath": filepath,
            "detections": detections,
        }


class SpeciesNetEnsemble:
    """Ensemble component of SpeciesNet."""

    def __init__(self, model_name: str, geofence: bool = True) -> None:
        """Loads the ensemble resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
            geofence:
                Whether to enable geofencing. If `False` skip it entirely.
        """

        start_time = time.time()

        self.model_info = ModelInfo(model_name)
        self.enable_geofence = geofence

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        # Create taxonomy map.
        with open(self.model_info.taxonomy, mode="r", encoding="utf-8") as fp:
            labels = [line.strip() for line in fp.readlines()]
            self.taxonomy_map = {_taxa_from_label(label): label for label in labels}

            for label in [
                Classification.BLANK,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                taxa = _taxa_from_label(label)
                if taxa in self.taxonomy_map:
                    del self.taxonomy_map[taxa]

            for label in [Classification.HUMAN, Classification.ANIMAL]:
                taxa = _taxa_from_label(label)
                self.taxonomy_map[taxa] = label

        # Load geofence map.
        with open(self.model_info.geofence, mode="r", encoding="utf-8") as fp:
            self.geofence_map = json.load(fp)

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetEnsemble in %s.",
            format_timespan(end_time - start_time),
        )

    def _get_ancestor_at_level(self, label: str, taxonomy_level: str) -> Optional[str]:
        """Finds the taxonomy item corresponding to a label's ancestor at a given level.

        E.g. The ancestor at family level for
        `uuid;class;order;family;genus;species;common_name` is
        `another_uuid;class;order;family;;;another_common_name`.

        Args:
            label:
                String label for which to find the ancestor.
            taxonomy_level:
                One of "species", "genus", "family", "order", "class" or "kingdom",
                indicating the taxonomy level at which to find a label's ancestor.

        Returns:
            A string label indicating the ancestor at the requested taxonomy level. In
            case the taxonomy doesn't contain the corresponding ancestor, return `None`.

        Raises:
            ValueError:
                If the given label is invalid.
        """

        label_parts = label.split(";")
        if len(label_parts) != 7:
            raise ValueError(
                f"Expected label made of 7 parts, but found only {len(label_parts)}: "
                f"{label}"
            )

        if taxonomy_level == "species":
            ancestor_parts = label_parts[1:6]
            if not ancestor_parts[4]:
                return None
        elif taxonomy_level == "genus":
            ancestor_parts = label_parts[1:5] + [""]
            if not ancestor_parts[3]:
                return None
        elif taxonomy_level == "family":
            ancestor_parts = label_parts[1:4] + ["", ""]
            if not ancestor_parts[2]:
                return None
        elif taxonomy_level == "order":
            ancestor_parts = label_parts[1:3] + ["", "", ""]
            if not ancestor_parts[1]:
                return None
        elif taxonomy_level == "class":
            ancestor_parts = label_parts[1:2] + ["", "", "", ""]
            if not ancestor_parts[0]:
                return None
        elif taxonomy_level == "kingdom":
            ancestor_parts = ["", "", "", "", ""]
            if not label_parts[1] and label != Classification.ANIMAL:
                return None
        else:
            return None

        ancestor = ";".join(ancestor_parts)
        return self.taxonomy_map.get(ancestor)

    def _get_full_class_string(self, label: str) -> str:
        """Extracts the full class string corresponding to a given label.

        E.g. The full class string for the label
        `uuid;class;order;family;genus;species;common_name` is
        `class;order;family;genus;species`.

        Args:
            label:
                String label for which to extract the full class string.

        Returns:
            Full class string for the given label.

        Raises:
            ValueError: If the given label is invalid.
        """

        label_parts = label.split(";")
        if len(label_parts) != 7:
            raise ValueError(
                f"Expected label made of 7 parts, but found only {len(label_parts)}: "
                f"{label}"
            )
        return ";".join(label_parts[1:6])

    def _should_geofence_animal_classification(
        self,
        label: str,
        country: Optional[str],
        admin1_region: Optional[str],
    ) -> bool:
        """Checks whether to geofence animal prediction in a country or admin1_region.

        Args:
            label:
                Animal label to check geofence rules for.
            country:
                Country (in ISO 3166-1 alpha-3 format) to check geofence rules for.
                Optional.
            admin1_region:
                First-level administrative division (in ISO 3166-2 format) to check
                geofence rules for. Optional.

        Returns:
            A boolean indicating whether to geofence given animal prediction.
        """

        # Do not geofence if geofencing is disabled.
        if not self.enable_geofence:
            return False

        # Do not geofence if country was not provided.
        if not country:
            return False

        # Do not geofence if full class string is missing from the geofence map.
        full_class_string = self._get_full_class_string(label)
        if full_class_string not in self.geofence_map:
            return False

        # Check if we need to geofence based on "allow" rules.
        allow_countries = self.geofence_map[full_class_string].get("allow")
        if allow_countries:
            if country not in allow_countries:
                # Geofence when country was not explicitly allowed.
                return True
            else:
                allow_admin1_regions = allow_countries[country]
                if (
                    admin1_region
                    and allow_admin1_regions
                    and admin1_region not in allow_admin1_regions
                ):
                    # Geofence when admin1_region was not explicitly allowed.
                    return True

        # Check if we need to geofence based on "block" rules.
        block_countries = self.geofence_map[full_class_string].get("block")
        if block_countries:
            if country in block_countries:
                block_admin1_regions = block_countries[country]
                if not block_admin1_regions:
                    # Geofence when entire country was blocked.
                    return True
                elif admin1_region and admin1_region in block_admin1_regions:
                    # Geofence when admin1_region was blocked.
                    return True

        # Do not geofence if no rule enforced that.
        return False

    def _roll_up_labels_to_first_matching_level(
        self,
        labels: list[str],
        scores: list[float],
        country: Optional[str],
        admin1_region: Optional[str],
        target_taxonomy_levels: list[str],
        non_blank_threshold: float,
    ) -> Optional[PredictionType]:
        """Rolls up prediction labels to the first taxonomy level above given threshold.

        Args:
            labels:
                List of classification labels.
            scores:
                List of classification scores.
            country:
                Country (in ISO 3166-1 alpha-3 format) associated with prediction.
                Optional.
            admin1_region:
                First-level administrative division (in ISO 3166-2 format) associated
                with prediction. Optional.
            target_taxonomy_levels:
                Ordered list of taxonomy levels at which to roll up classification
                labels and check if the cumulative score passes the given threshold.
                Levels must be a subset of: "species", "genus", "family", "order",
                "class", "kingdom".
            non_blank_threshold:
                Min threshold at which the cumulative score is good enough to consider
                the rollup successful.

        Returns:
            A tuple of <label, score, prediction_source> describing the first taxonomy
            level at which the cumulative score passes the given threshold. If no such
            level exists, return `None`.

        Raises:
            ValueError:
                If the taxonomy level if not one of: "species", "genus", "family",
                "order", "class", "kingdom".
        """

        expected_target_taxonomy_levels = {
            "species",
            "genus",
            "family",
            "order",
            "class",
            "kingdom",
        }
        unknown_target_taxonomy_levels = set(target_taxonomy_levels).difference(
            expected_target_taxonomy_levels
        )
        if unknown_target_taxonomy_levels:
            raise ValueError(
                "Unexpected target taxonomy level(s): "
                f"{unknown_target_taxonomy_levels}. "
                f"Expected only levels from the set: {expected_target_taxonomy_levels}."
            )

        # Accumulate scores at each taxonomy level and, if they pass the desired
        # threshold, return that rollup label.
        for taxonomy_level in target_taxonomy_levels:
            accumulated_scores = {}
            for label, score in zip(labels, scores):
                rollup_label = self._get_ancestor_at_level(label, taxonomy_level)
                if rollup_label:
                    new_score = accumulated_scores.get(rollup_label, 0.0) + score
                    accumulated_scores[rollup_label] = new_score

            max_rollup_label = None
            max_rollup_score = 0.0
            for rollup_label, rollup_score in accumulated_scores.items():
                if (
                    rollup_score > max_rollup_score
                    and not self._should_geofence_animal_classification(
                        rollup_label, country, admin1_region
                    )
                ):
                    max_rollup_label = rollup_label
                    max_rollup_score = rollup_score
            if max_rollup_score > non_blank_threshold and max_rollup_label:
                return (
                    max_rollup_label,
                    max_rollup_score,
                    f"classifier+rollup_to_{taxonomy_level}",
                )

        return None

    def _geofence_animal_classification(
        self,
        labels: list[str],
        scores: list[float],
        country: Optional[str],
        admin1_region: Optional[str],
    ) -> PredictionType:
        """Geofences animal prediction in a country or admin1_region.

        Under the hood, this also rolls up the labels every time it encounters a
        geofenced label.

        Args:
            labels:
                List of classification labels.
            scores:
                List of classification scores.
            country:
                Country (in ISO 3166-1 alpha-3 format) associated with prediction.
                Optional.
            admin1_region:
                First-level administrative division (in ISO 3166-2 format) associated
                with prediction. Optional.

        Returns:
            A tuple of <label, score, prediction_source> describing the result of the
            combined geofence and rollup operations.
        """

        if self._should_geofence_animal_classification(
            labels[0], country, admin1_region
        ):
            rollup = self._roll_up_labels_to_first_matching_level(
                labels=labels,
                scores=scores,
                country=country,
                admin1_region=admin1_region,
                target_taxonomy_levels=["family", "order", "class", "kingdom"],
                # Force the rollup to pass the top classification score.
                non_blank_threshold=scores[0] - 1e-10,
            )
            if rollup:
                rollup_label, rollup_score, rollup_source = rollup
                return (
                    rollup_label,
                    rollup_score,
                    f"classifier+geofence+{rollup_source[len('classifier+'):]}",
                )
            else:
                # Normally, this return statement should never be reached since the
                # animal rollup would eventually succeed (even though that may be at
                # "kingdom" level, as a last resort). The only scenario when this could
                # still be reached is if the method was incorrectly called with a list
                # of non-animal labels (e.g. blanks, vehicles). In this case it's best
                # to return an unknown classification, while propagating the top score.
                return (
                    Classification.UNKNOWN,
                    scores[0],
                    "classifier+geofence+rollup_failed",
                )
        else:
            return labels[0], scores[0], "classifier"

    def _combine_predictions_for_single_item(
        self,
        classifications: dict[str, list],
        detections: list[dict],
        country: Optional[str],
        admin1_region: Optional[str],
    ) -> PredictionType:
        """Ensembles classifications and detections for a single image.

        This operation leverages multiple heuristics to make the most of the classifier
        and the detector predictions through a complex set of decisions. It introduces
        various thresholds to identify humans, vehicles, blanks, animals at species
        level, animals at higher taxonomy levels and even unknowns.

        Args:
            classifications:
                Dict of classification results. "classes" and "scores" are expected to
                be provided among the dict keys.
            detections:
                List of detection results, sorted in decreasing order of their
                confidence score. Each detection is expected to be a dict providing
                "label" and "conf" among its keys.
            country:
                Country (in ISO 3166-1 alpha-3 format) associated with predictions.
                Optional.
            admin1_region:
                First-level administrative division (in ISO 3166-2 format) associated
                with predictions. Optional.

        Returns:
            A tuple of <label, score, prediction_source> describing the ensemble result.
        """

        top_classification_class = classifications["classes"][0]
        top_classification_score = classifications["scores"][0]
        top_detection_class = detections[0]["label"] if detections else Detection.ANIMAL
        top_detection_score = detections[0]["conf"] if detections else 0.0

        if top_detection_class == Detection.HUMAN:
            # Threshold #1a: high-confidence HUMAN detections.
            if top_detection_score > 0.7:
                return Classification.HUMAN, top_detection_score, "detector"

            # Threshold #1b: mid-confidence HUMAN detections + high-confidence
            # HUMAN/VEHICLE classifications.
            if (
                top_detection_score > 0.2
                and top_classification_class
                in {Classification.HUMAN, Classification.VEHICLE}
                and top_classification_score > 0.5
            ):
                return Classification.HUMAN, top_classification_score, "classifier"

        if top_detection_class == Detection.VEHICLE:
            # Threshold #2a: mid-confidence VEHICLE detections + high-confidence HUMAN
            # classifications.
            if (
                top_detection_score > 0.2
                and top_classification_class == Classification.HUMAN
                and top_classification_score > 0.5
            ):
                return Classification.HUMAN, top_classification_score, "classifier"

            # Threshold #2b: high-confidence VEHICLE detections.
            if top_detection_score > 0.7:
                return Classification.VEHICLE, top_detection_score, "detector"

            # Threshold #2c: mid-confidence VEHICLE detections + high-confidence VEHICLE
            # classifications.
            if (
                top_detection_score > 0.2
                and top_classification_class == Classification.VEHICLE
                and top_classification_score > 0.4
            ):
                return Classification.VEHICLE, top_classification_score, "classifier"

        # Threshold #3a: high-confidence BLANK "detections" + high-confidence BLANK
        # classifications.
        if (
            top_detection_score < 0.2
            and top_classification_class == Classification.BLANK
            and top_classification_score > 0.5
        ):
            return Classification.BLANK, top_classification_score, "classifier"

        # Threshold #3b: extra-high-confidence BLANK classifications.
        if (
            top_classification_class == Classification.BLANK
            and top_classification_score > 0.99
        ):
            return Classification.BLANK, top_classification_score, "classifier"

        if top_classification_class not in {
            Classification.BLANK,
            Classification.HUMAN,
            Classification.VEHICLE,
        }:
            # Threshold #4a: extra-high-confidence ANIMAL classifications.
            if top_classification_score > 0.8:
                return self._geofence_animal_classification(
                    labels=classifications["classes"],
                    scores=classifications["scores"],
                    country=country,
                    admin1_region=admin1_region,
                )

            # Threshold #4b: high-confidence ANIMAL classifications + mid-confidence
            # ANIMAL detections.
            if (
                top_classification_score > 0.65
                and top_detection_class == Detection.ANIMAL
                and top_detection_score > 0.2
            ):
                return self._geofence_animal_classification(
                    labels=classifications["classes"],
                    scores=classifications["scores"],
                    country=country,
                    admin1_region=admin1_region,
                )

        # Threshold #5a: high-confidence ANIMAL rollups.
        rollup = self._roll_up_labels_to_first_matching_level(
            labels=classifications["classes"],
            scores=classifications["scores"],
            country=country,
            admin1_region=admin1_region,
            target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"],
            non_blank_threshold=0.65,
        )
        if rollup:
            return rollup

        # Threshold #5b: mid-confidence ANIMAL detections.
        if top_detection_class == Detection.ANIMAL and top_detection_score > 0.5:
            return Classification.ANIMAL, top_detection_score, "detector"

        return Classification.UNKNOWN, top_classification_score, "classifier"

    def combine(
        self,
        filepaths: list[str],
        classifier_results: dict[str, Any],
        detector_results: dict[str, Any],
        geolocation_results: dict[str, Any],
        exif_results: dict[str, PIL.Image.Exif],
        partial_predictions: dict[str, dict],
    ) -> list[dict[str, Any]]:
        """Ensembles classifications and detections for a list of images.

        Args:
            filepaths:
                List of filepaths to ensemble predictions for.
            classifier_results:
                Dict of classifier results, with keys given by the filepaths to ensemble
                predictions for.
            detector_results:
                Dict of detector results, with keys given by the filepaths to ensemble
                predictions for.
            geolocation_results:
                Dict of geolocation results, with keys given by the filepaths to
                ensemble predictions for.
            exif_results:
                Dict of EXIF results, with keys given by the filepaths to ensemble
                predictions for.
            partial_predictions:
                Dict of partial predictions from previous ensemblings, with keys given
                by the filepaths for which predictions where already ensembled. Used to
                skip re-ensembling for the matching filepaths.

        Returns:
            List of ensembled predictions.
        """

        results = []
        for filepath in filepaths:
            # Use the result from previously computed predictions when available.
            if filepath in partial_predictions:
                results.append(partial_predictions[filepath])
                continue

            # Check for failures.
            failure = Failure(0)
            if (
                filepath in classifier_results
                and "failure" not in classifier_results[filepath]
            ):
                classifications = classifier_results[filepath]["classifications"]
            else:
                classifications = None
                failure |= Failure.CLASSIFIER
            if (
                filepath in detector_results
                and "failure" not in detector_results[filepath]
            ):
                detections = detector_results[filepath]["detections"]
            else:
                detections = None
                failure |= Failure.DETECTOR
            if filepath in geolocation_results:
                geolocation = geolocation_results[filepath]
            else:
                geolocation = {}
                failure |= Failure.GEOLOCATION

            # Add as much raw information as possible to the prediction result.
            result = {
                "filepath": filepath,
                "failures": (
                    [f.name for f in Failure if f in failure] if failure else None
                ),
                "country": geolocation.get("country"),
                "admin1_region": geolocation.get("admin1_region"),
                "latitude": geolocation.get("latitude"),
                "longitude": geolocation.get("longitude"),
                "classifications": classifications,
                "detections": detections,
            }
            result = {key: value for key, value in result.items() if value is not None}

            # Most importantly, ensemble everything into a single prediction.
            if classifications is not None and detections is not None:
                prediction, score, source = self._combine_predictions_for_single_item(
                    classifications=classifications,
                    detections=detections,
                    country=geolocation.get("country"),
                    admin1_region=geolocation.get("admin1_region"),
                )
                result["prediction"] = (
                    prediction.value
                    if isinstance(prediction, Classification)
                    else prediction
                )
                result["prediction_score"] = score
                result["prediction_source"] = source

            # Add EXIF information.
            exif = exif_results.get(filepath)
            if exif:
                selected_exif = {}
                date_time_original = exif.get_ifd(PIL.ExifTags.IFD.Exif).get(
                    PIL.ExifTags.Base.DateTimeOriginal
                )
                if date_time_original:
                    selected_exif["DateTimeOriginal"] = date_time_original
                if selected_exif:
                    result["exif"] = selected_exif

            # Finally, report the model version.
            result["model_version"] = self.model_info.version

            results.append(result)

        return results
