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

"""Logic for combining predictions for the SpeciesNet ensemble."""

from typing import Callable, Optional

from speciesnet.constants import Classification
from speciesnet.constants import Detection

PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


def combine_predictions_for_single_item(
    *,
    classifications: dict[str, list],
    detections: list[dict],
    country: Optional[str],
    admin1_region: Optional[str],
    taxonomy_map: dict,
    geofence_map: dict,
    enable_geofence: bool,
    geofence_fn: Callable,
    roll_up_fn: Callable,
) -> PredictionType:
    """Ensembles classifications and detections for a single image.

    This operation leverages multiple heuristics to make the most of the classifier and
    the detector predictions through a complex set of decisions. It introduces various
    thresholds to identify humans, vehicles, blanks, animals at species level, animals
    at higher taxonomy levels and even unknowns.

    Args:
        classifications:
            Dict of classification results. "classes" and "scores" are expected to be
            provided among the dict keys.
        detections:
            List of detection results, sorted in decreasing order of their confidence
            score. Each detection is expected to be a dict providing "label" and "conf"
            among its keys.
        country:
            Country (in ISO 3166-1 alpha-3 format) associated with predictions.
            Optional.
        admin1_region:
            First-level administrative division (in ISO 3166-2 format) associated with
            predictions. Optional.
        taxonomy_map:
            Dictionary mapping taxa to labels.
        geofence_map:
            Dictionary mapping full class strings to geofence rules.
        enable_geofence:
            Whether geofencing is enabled.
        geofence_fn:
            Callable to geofence animal classifications.
        roll_up_fn:
            Callable to roll up labels to the first matching level.

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
            return geofence_fn(
                labels=classifications["classes"],
                scores=classifications["scores"],
                country=country,
                admin1_region=admin1_region,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=enable_geofence,
            )

        # Threshold #4b: high-confidence ANIMAL classifications + mid-confidence
        # ANIMAL detections.
        if (
            top_classification_score > 0.65
            and top_detection_class == Detection.ANIMAL
            and top_detection_score > 0.2
        ):
            return geofence_fn(
                labels=classifications["classes"],
                scores=classifications["scores"],
                country=country,
                admin1_region=admin1_region,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=enable_geofence,
            )

    # Threshold #5a: high-confidence ANIMAL rollups.
    rollup = roll_up_fn(
        labels=classifications["classes"],
        scores=classifications["scores"],
        country=country,
        admin1_region=admin1_region,
        target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"],
        non_blank_threshold=0.65,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=enable_geofence,
    )
    if rollup:
        return rollup

    # Threshold #5b: mid-confidence ANIMAL detections.
    if top_detection_class == Detection.ANIMAL and top_detection_score > 0.5:
        return Classification.ANIMAL, top_detection_score, "detector"

    return Classification.UNKNOWN, top_classification_score, "classifier"
