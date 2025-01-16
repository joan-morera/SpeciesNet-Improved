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

"""Ensemble functionality of SpeciesNet."""

__all__ = [
    "SpeciesNetEnsemble",
]

import json
import time
from typing import Any, Optional

from absl import logging
from humanfriendly import format_timespan
import PIL.ExifTags
import PIL.Image

from speciesnet.constants import Classification
from speciesnet.constants import Detection
from speciesnet.constants import Failure
from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
from speciesnet.utils import ModelInfo

# Handy type aliases.
PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


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
        self.taxonomy_map = self.load_taxonomy()
        self.geofence_map = self.load_geofence()

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetEnsemble in %s.",
            format_timespan(end_time - start_time),
        )

    def load_taxonomy(self):
        """Loads the taxonomy from the model info."""

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        # Create taxonomy map.
        with open(self.model_info.taxonomy, mode="r", encoding="utf-8") as fp:
            labels = [line.strip() for line in fp.readlines()]
            taxonomy_map = {_taxa_from_label(label): label for label in labels}

            for label in [
                Classification.BLANK,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                taxa = _taxa_from_label(label)
                if taxa in taxonomy_map:
                    del taxonomy_map[taxa]

            for label in [Classification.HUMAN, Classification.ANIMAL]:
                taxa = _taxa_from_label(label)
                taxonomy_map[taxa] = label
        return taxonomy_map

    def load_geofence(self):
        """Loads the geofence map from the model info."""

        with open(self.model_info.geofence, mode="r", encoding="utf-8") as fp:
            geofence_map = json.load(fp)
        return geofence_map

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
                return geofence_animal_classification(
                    labels=classifications["classes"],
                    scores=classifications["scores"],
                    country=country,
                    admin1_region=admin1_region,
                    taxonomy_map=self.taxonomy_map,
                    geofence_map=self.geofence_map,
                    enable_geofence=self.enable_geofence,
                )

            # Threshold #4b: high-confidence ANIMAL classifications + mid-confidence
            # ANIMAL detections.
            if (
                top_classification_score > 0.65
                and top_detection_class == Detection.ANIMAL
                and top_detection_score > 0.2
            ):
                return geofence_animal_classification(
                    labels=classifications["classes"],
                    scores=classifications["scores"],
                    country=country,
                    admin1_region=admin1_region,
                    taxonomy_map=self.taxonomy_map,
                    geofence_map=self.geofence_map,
                    enable_geofence=self.enable_geofence,
                )

        # Threshold #5a: high-confidence ANIMAL rollups.
        rollup = roll_up_labels_to_first_matching_level(
            labels=classifications["classes"],
            scores=classifications["scores"],
            country=country,
            admin1_region=admin1_region,
            target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"],
            non_blank_threshold=0.65,
            taxonomy_map=self.taxonomy_map,
            geofence_map=self.geofence_map,
            enable_geofence=self.enable_geofence,
        )
        if rollup:
            return rollup

        # Threshold #5b: mid-confidence ANIMAL detections.
        if top_detection_class == Detection.ANIMAL and top_detection_score > 0.5:
            return Classification.ANIMAL, top_detection_score, "detector"

        return Classification.UNKNOWN, top_classification_score, "classifier"

    def combine(  # pylint: disable=too-many-positional-arguments
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
