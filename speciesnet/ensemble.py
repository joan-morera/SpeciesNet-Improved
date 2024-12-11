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

    def _roll_up_labels_to_first_matching_level(  # pylint: disable=too-many-positional-arguments
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
