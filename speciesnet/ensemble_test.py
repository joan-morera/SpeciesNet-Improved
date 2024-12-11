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

# pylint: disable=missing-module-docstring
# pylint: disable=protected-access

import warnings

import pytest

from speciesnet.constants import Classification
from speciesnet.ensemble import PredictionType
from speciesnet.ensemble import SpeciesNetEnsemble
from speciesnet.utils import load_rgb_image

# fmt: off
# pylint: disable=line-too-long

BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
BLANK_FC = ";;;;"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
HUMAN_FC = "mammalia;primates;hominidae;homo;sapiens"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
VEHICLE_FC = ";;;;"

LION = "ddf59264-185a-4d35-b647-2785792bdf54;mammalia;carnivora;felidae;panthera;leo;lion"
LION_FC = "mammalia;carnivora;felidae;panthera;leo"
PANTHERA_GENUS = "fbb23d07-6677-43db-b650-f99ac452c50f;mammalia;carnivora;felidae;panthera;;panthera species"
PANTHERA_GENUS_FC = "mammalia;carnivora;felidae;panthera;"
FELIDAE_FAMILY = "df8514b0-10a5-411f-8ed6-0f415e8153a3;mammalia;carnivora;felidae;;;cat family"
FELIDAE_FAMILY_FC = "mammalia;carnivora;felidae;;"
CARNIVORA_ORDER = "eeeb5d26-2a47-4d01-a3de-10b33ec0aee4;mammalia;carnivora;;;;carnivorous mammal"
CARNIVORA_ORDER_FC = "mammalia;carnivora;;;"
MAMMALIA_CLASS = "f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal"
MAMMALIA_CLASS_FC = "mammalia;;;;"
ANIMAL_KINGDOM = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
ANIMAL_KINGDOM_FC = ";;;;"

BROWN_BEAR = "330bb1e9-84d6-4e41-afa9-938aee17ea29;mammalia;carnivora;ursidae;ursus;arctos;brown bear"
BROWN_BEAR_FC = "mammalia;carnivora;ursidae;ursus;arctos"
POLAR_BEAR = "e7f83bf6-df2c-4ce0-97fc-2f233df23ec4;mammalia;carnivora;ursidae;ursus;maritimus;polar bear"
POLAR_BEAR_FC = "mammalia;carnivora;ursidae;ursus;maritimus"
GIANT_PANDA = "85662682-67c1-4ecb-ba05-ba12e2df6b65;mammalia;carnivora;ursidae;ailuropoda;melanoleuca;giant panda"
GIANT_PANDA_FC = "mammalia;carnivora;ursidae;ailuropoda;melanoleuca"
URSUS_GENUS = "5a0f5e3f-c634-4b86-910a-b105cb526a24;mammalia;carnivora;ursidae;ursus;;ursus species"
URSUS_GENUS_FC = "mammalia;carnivora;ursidae;ursus;"
URSIDAE_FAMILY = "ec1a70f4-41c0-4aba-9150-292fb2b7a324;mammalia;carnivora;ursidae;;;bear family"
URSIDAE_FAMILY_FC = "mammalia;carnivora;ursidae;;"

PUMA = "9c564562-9429-405c-8529-04cff7752282;mammalia;carnivora;felidae;puma;concolor;puma"
PUMA_FC = "mammalia;carnivora;felidae;puma;concolor"
SAND_CAT = "e588253d-d61d-4149-a96c-8c245927a80f;mammalia;carnivora;felidae;felis;margarita;sand cat"
SAND_CAT_FC = "mammalia;carnivora;felidae;felis;margarita"

# pylint: enable=line-too-long
# fmt: on


class TestEnsemble:
    """Tests for the ensemble component."""

    @pytest.fixture(scope="class")
    def ensemble(self, model_name: str) -> SpeciesNetEnsemble:
        return SpeciesNetEnsemble(model_name)

    @pytest.fixture
    def mock_ensemble(self, monkeypatch, ensemble) -> SpeciesNetEnsemble:
        taxonomy_map = {
            BLANK_FC: BLANK,
            HUMAN_FC: HUMAN,
            VEHICLE_FC: VEHICLE,
            LION_FC: LION,
            PANTHERA_GENUS_FC: PANTHERA_GENUS,
            FELIDAE_FAMILY_FC: FELIDAE_FAMILY,
            CARNIVORA_ORDER_FC: CARNIVORA_ORDER,
            MAMMALIA_CLASS_FC: MAMMALIA_CLASS,
            ANIMAL_KINGDOM_FC: ANIMAL_KINGDOM,
            BROWN_BEAR_FC: BROWN_BEAR,
            POLAR_BEAR_FC: POLAR_BEAR,
            GIANT_PANDA_FC: GIANT_PANDA,
            URSUS_GENUS_FC: URSUS_GENUS,
            URSIDAE_FAMILY_FC: URSIDAE_FAMILY,
        }
        monkeypatch.setattr(ensemble, "taxonomy_map", taxonomy_map)

        geofence_map = {
            LION_FC: {
                "allow": {
                    "KEN": [],
                    "TZA": [],
                }
            },
            PANTHERA_GENUS_FC: {
                "allow": {
                    "KEN": [],
                    "TZA": [],
                    "USA": ["AK", "CA"],
                }
            },
            FELIDAE_FAMILY_FC: {
                "allow": {
                    "FRA": [],
                    "KEN": [],
                    "TZA": [],
                    "USA": [],
                },
                "block": {
                    "FRA": [],
                    "USA": ["NY"],
                },
            },
            SAND_CAT_FC: {
                "block": {
                    "AUS": [],
                },
            },
            URSIDAE_FAMILY_FC: {
                "block": {
                    "GBR": [],
                },
            },
        }
        monkeypatch.setattr(ensemble, "geofence_map", geofence_map)

        return ensemble

    @pytest.fixture
    def mock_ensemble_no_geofence(
        self, monkeypatch, mock_ensemble
    ) -> SpeciesNetEnsemble:
        monkeypatch.setattr(mock_ensemble, "enable_geofence", False)
        return mock_ensemble

    @pytest.fixture
    def mock_ensemble2(self, monkeypatch, ensemble) -> SpeciesNetEnsemble:

        def _combine_predictions_for_single_item(
            classifications: dict[str, list], *args, **kwargs
        ) -> PredictionType:
            del args  # Unused.
            del kwargs  # Unused.
            return classifications["classes"][0], classifications["scores"][0], "mock"

        monkeypatch.setattr(
            ensemble,
            "_combine_predictions_for_single_item",
            _combine_predictions_for_single_item,
        )
        return ensemble

    def test_get_ancestor_at_level(self, mock_ensemble) -> None:

        # Test all ancestors of LION.
        assert mock_ensemble._get_ancestor_at_level(LION, "species") == LION
        assert mock_ensemble._get_ancestor_at_level(LION, "genus") == PANTHERA_GENUS
        assert mock_ensemble._get_ancestor_at_level(LION, "family") == FELIDAE_FAMILY
        assert mock_ensemble._get_ancestor_at_level(LION, "order") == CARNIVORA_ORDER
        assert mock_ensemble._get_ancestor_at_level(LION, "class") == MAMMALIA_CLASS
        assert mock_ensemble._get_ancestor_at_level(LION, "kingdom") == ANIMAL_KINGDOM

        # Test all ancestors of PANTHERA_GENUS.
        assert mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "species") is None
        assert (
            mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "genus")
            == PANTHERA_GENUS
        )
        assert (
            mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "family")
            == FELIDAE_FAMILY
        )
        assert (
            mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "order")
            == CARNIVORA_ORDER
        )
        assert (
            mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "class")
            == MAMMALIA_CLASS
        )
        assert (
            mock_ensemble._get_ancestor_at_level(PANTHERA_GENUS, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of FELIDAE_FAMILY.
        assert mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "species") is None
        assert mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "genus") is None
        assert (
            mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "family")
            == FELIDAE_FAMILY
        )
        assert (
            mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "order")
            == CARNIVORA_ORDER
        )
        assert (
            mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "class")
            == MAMMALIA_CLASS
        )
        assert (
            mock_ensemble._get_ancestor_at_level(FELIDAE_FAMILY, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of CARNIVORA_ORDER.
        assert mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "species") is None
        assert mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "family") is None
        assert (
            mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "order")
            == CARNIVORA_ORDER
        )
        assert (
            mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "class")
            == MAMMALIA_CLASS
        )
        assert (
            mock_ensemble._get_ancestor_at_level(CARNIVORA_ORDER, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of MAMMALIA_CLASS.
        assert mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "species") is None
        assert mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "family") is None
        assert mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "order") is None
        assert (
            mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "class")
            == MAMMALIA_CLASS
        )
        assert (
            mock_ensemble._get_ancestor_at_level(MAMMALIA_CLASS, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of ANIMAL_KINGDOM.
        assert mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "species") is None
        assert mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "family") is None
        assert mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "order") is None
        assert mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "class") is None
        assert (
            mock_ensemble._get_ancestor_at_level(ANIMAL_KINGDOM, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of BLANK.
        assert mock_ensemble._get_ancestor_at_level(BLANK, "species") is None
        assert mock_ensemble._get_ancestor_at_level(BLANK, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(BLANK, "family") is None
        assert mock_ensemble._get_ancestor_at_level(BLANK, "order") is None
        assert mock_ensemble._get_ancestor_at_level(BLANK, "class") is None
        assert mock_ensemble._get_ancestor_at_level(BLANK, "kingdom") is None

        # Test all ancestors of HUMAN, when its genus, family and order are missing from
        # the mock taxonomy mapping.
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "species") == HUMAN
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "family") is None
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "order") is None
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "class") == MAMMALIA_CLASS
        assert mock_ensemble._get_ancestor_at_level(HUMAN, "kingdom") == ANIMAL_KINGDOM

        # Test all ancestors of VEHICLE.
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "species") is None
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "family") is None
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "order") is None
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "class") is None
        assert mock_ensemble._get_ancestor_at_level(VEHICLE, "kingdom") is None

        # Test all ancestors of an unseen species.
        unseen_species = "uuid;class;order;family;genus;species;common_name"
        assert mock_ensemble._get_ancestor_at_level(unseen_species, "species") is None
        assert mock_ensemble._get_ancestor_at_level(unseen_species, "genus") is None
        assert mock_ensemble._get_ancestor_at_level(unseen_species, "family") is None
        assert mock_ensemble._get_ancestor_at_level(unseen_species, "order") is None
        assert mock_ensemble._get_ancestor_at_level(unseen_species, "class") is None
        assert (
            mock_ensemble._get_ancestor_at_level(unseen_species, "kingdom")
            == ANIMAL_KINGDOM
        )

        # Test errors due to invalid labels.
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species"
            mock_ensemble._get_ancestor_at_level(invalid_label, "kingdom")
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species;common_name;extra"
            mock_ensemble._get_ancestor_at_level(invalid_label, "kingdom")

    def test_get_full_class_string(self, mock_ensemble) -> None:

        # Test BLANK/HUMAN/VEHICLE.
        assert mock_ensemble._get_full_class_string(BLANK) == BLANK_FC
        assert mock_ensemble._get_full_class_string(HUMAN) == HUMAN_FC
        assert mock_ensemble._get_full_class_string(VEHICLE) == VEHICLE_FC

        # Test valid labels at different taxonomy levels.
        assert mock_ensemble._get_full_class_string(LION) == LION_FC
        assert mock_ensemble._get_full_class_string(PANTHERA_GENUS) == PANTHERA_GENUS_FC
        assert mock_ensemble._get_full_class_string(FELIDAE_FAMILY) == FELIDAE_FAMILY_FC
        assert (
            mock_ensemble._get_full_class_string(CARNIVORA_ORDER) == CARNIVORA_ORDER_FC
        )
        assert mock_ensemble._get_full_class_string(MAMMALIA_CLASS) == MAMMALIA_CLASS_FC
        assert mock_ensemble._get_full_class_string(ANIMAL_KINGDOM) == ANIMAL_KINGDOM_FC

        # Test errors due to invalid labels.
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species"
            mock_ensemble._get_full_class_string(invalid_label)
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species;common_name;extra"
            mock_ensemble._get_full_class_string(invalid_label)

    def test_should_geofence_animal_classification(self, mock_ensemble) -> None:

        # Test when country is not provided.
        assert not mock_ensemble._should_geofence_animal_classification(
            LION, country=None, admin1_region=None
        )

        # Test when label is not in the geofence map.
        assert not mock_ensemble._should_geofence_animal_classification(
            PUMA, country="USA", admin1_region=None
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            PUMA, country="USA", admin1_region="CA"
        )

        # Test "allow" rules from the geofence map.
        assert mock_ensemble._should_geofence_animal_classification(
            LION, country="GBR", admin1_region=None
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            LION, country="KEN", admin1_region=None
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region=None
        )
        assert mock_ensemble._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region="NY"
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region="CA"
        )

        # Test "block" rules from the geofence map.
        assert mock_ensemble._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="FRA", admin1_region=None
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="TZA", admin1_region=None
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="USA", admin1_region="CA"
        )
        assert mock_ensemble._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="USA", admin1_region="NY"
        )
        assert not mock_ensemble._should_geofence_animal_classification(
            SAND_CAT, country="GBR", admin1_region=None
        )
        assert mock_ensemble._should_geofence_animal_classification(
            SAND_CAT, country="AUS", admin1_region=None
        )

    def test_should_geofence_animal_classification_disabled(
        self, mock_ensemble_no_geofence
    ) -> None:

        # Test when country is not provided.
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            LION, country=None, admin1_region=None
        )

        # Test when label is not in the geofence map.
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            PUMA, country="USA", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            PUMA, country="USA", admin1_region="CA"
        )

        # Test "allow" rules from the geofence map.
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            LION, country="GBR", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            LION, country="KEN", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region="NY"
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            PANTHERA_GENUS, country="USA", admin1_region="CA"
        )

        # Test "block" rules from the geofence map.
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="FRA", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="TZA", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="USA", admin1_region="CA"
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            FELIDAE_FAMILY, country="USA", admin1_region="NY"
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            SAND_CAT, country="GBR", admin1_region=None
        )
        assert not mock_ensemble_no_geofence._should_geofence_animal_classification(
            SAND_CAT, country="AUS", admin1_region=None
        )

    def test_roll_up_labels_to_first_matching_level(self, mock_ensemble) -> None:
        # pylint: disable=unnecessary-lambda-assignment

        predictions = [
            BROWN_BEAR,
            POLAR_BEAR,
            GIANT_PANDA,
            BLANK,
            LION,
            HUMAN,
            ANIMAL_KINGDOM,
        ]

        # Test rollups to species level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["species"],
                non_blank_threshold=0.9,
            )
        )
        assert rollup_fn([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]) is None
        assert rollup_fn([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]) is None
        assert rollup_fn([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == (
            BROWN_BEAR,
            pytest.approx(1.0),
            "classifier+rollup_to_species",
        )

        # Test rollups to genus level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["genus"],
                non_blank_threshold=0.9,
            )
        )
        assert rollup_fn([0.6, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01]) is None
        assert rollup_fn([0.7, 0.25, 0.01, 0.01, 0.01, 0.01, 0.01]) == (
            URSUS_GENUS,
            pytest.approx(0.95),
            "classifier+rollup_to_genus",
        )

        # Test rollups to family level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["family"],
                non_blank_threshold=0.8,
            )
        )
        assert rollup_fn([0.4, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]) is None
        assert rollup_fn([0.4, 0.21, 0.2, 0.0, 0.0, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.81),
            "classifier+rollup_to_family",
        )

        # Test rollups to order level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["order"],
                non_blank_threshold=0.8,
            )
        )
        assert rollup_fn([0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]) is None
        assert rollup_fn([0.3, 0.2, 0.1, 0.1, 0.23, 0.0, 0.0]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.83),
            "classifier+rollup_to_order",
        )

        # Test rollups to class level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["class"],
                non_blank_threshold=0.8,
            )
        )
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0]) is None
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.22, 0.1, 0.0]) == (
            MAMMALIA_CLASS,
            pytest.approx(0.82),
            "classifier+rollup_to_class",
        )

        # Test rollups to kingdom level.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["kingdom"],
                non_blank_threshold=0.81,
            )
        )
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) is None
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.23, 0.1, 0.1]) == (
            ANIMAL_KINGDOM,
            pytest.approx(0.93),
            "classifier+rollup_to_kingdom",
        )

        # Test rollups when multiple taxonomy levels are specified.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"],
                non_blank_threshold=0.75,
            )
        )
        assert rollup_fn([0.6, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.8),
            "classifier+rollup_to_family",
        )

        # Test rollups when multiple score sums pass the non blank threshold.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["species"],
                non_blank_threshold=0.1,
            )
        )
        assert rollup_fn([0.2, 0.3, 0.15, 0.0, 0.35, 0.0, 0.0]) == (
            LION,
            pytest.approx(0.35),
            "classifier+rollup_to_species",
        )

        # Test rollups when the BLANK score dominates all the others.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["species", "genus", "family", "order", "class"],
                non_blank_threshold=0.4,
            )
        )
        assert rollup_fn([0.1, 0.2, 0.2, 0.45, 0.0, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.5),
            "classifier+rollup_to_family",
        )

        # Test rollups with geofencing.
        rollup_fn = (
            lambda scores: mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=scores,
                country="GBR",
                admin1_region=None,
                target_taxonomy_levels=["species", "genus", "family", "order", "class"],
                non_blank_threshold=0.4,
            )
        )
        assert rollup_fn([0.1, 0.2, 0.2, 0.45, 0.0, 0.0, 0.0]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.5),
            "classifier+rollup_to_order",
        )

        # Test rollups to invalid levels.
        with pytest.raises(ValueError):
            mock_ensemble._roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["invalid_level"],
                non_blank_threshold=0.3,
            )

    def test_geofence_animal_classification(self, mock_ensemble) -> None:
        # pylint: disable=unnecessary-lambda-assignment

        predictions = [
            LION,
            POLAR_BEAR,
            BLANK,
            FELIDAE_FAMILY,
        ]

        # Test when no geofencing is needed.
        geofence_fn = lambda scores: mock_ensemble._geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="TZA",
            admin1_region=None,
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            LION,
            pytest.approx(0.4),
            "classifier",
        )

        # Test with geofencing and rollup to family level or above.
        geofence_fn = lambda scores: mock_ensemble._geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="USA",
            admin1_region=None,
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            FELIDAE_FAMILY,
            pytest.approx(0.5),
            "classifier+geofence+rollup_to_family",
        )
        geofence_fn = lambda scores: mock_ensemble._geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="USA",
            admin1_region="NY",
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.8),
            "classifier+geofence+rollup_to_order",
        )

    def test_combine(self, mock_ensemble2) -> None:

        blank_img = load_rgb_image("test_data/blank.jpg")
        blank2_img = load_rgb_image("test_data/blank2.jpg")
        blank3_img = load_rgb_image("test_data/blank3.jpg")
        assert blank_img and blank2_img and blank3_img

        expected_model_version = mock_ensemble2.model_info.version

        filepaths = [
            "a.jpg",
            "b.jpg",
            "c.jpg",
            "d.jpg",
            "e.jpg",
            "f.jpg",
        ]
        classifier_results = {
            "b.jpg": {
                "filepath": "b.jpg",
                "failure": "classifier message",
            },
            "c.jpg": {
                "filepath": "c.jpg",
                "classifications": {
                    "classes": ["X", "Y", "Z"],
                    "scores": [0.5, 0.3, 0.2],
                },
            },
            "d.jpg": {
                "filepath": "d.jpg",
                "classifications": {
                    "classes": ["R", "S", "T"],
                    "scores": [0.7, 0.2, 0.1],
                },
            },
            "e.jpg": {
                "filepath": "e.jpg",
                "classifications": {
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                },
            },
            "f.jpg": {
                "filepath": "f.jpg",
                "classifications": {
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                },
            },
        }
        detector_results = {
            "a.jpg": {
                "filepath": "a.jpg",
                "failure": "detector message",
            },
            "b.jpg": {
                "filepath": "b.jpg",
                "detections": [
                    {
                        "category": "1",
                        "label": "animal",
                        "conf": 0.5,
                        "bbox": [0.0, 0.1, 0.2, 0.3],
                    }
                ],
            },
            "d.jpg": {
                "filepath": "d.jpg",
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            "e.jpg": {
                "filepath": "e.jpg",
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            "f.jpg": {
                "filepath": "f.jpg",
                "detections": [],
            },
        }
        geolocation_results = {
            "a.jpg": {
                "country": "COUNTRY_A",
            },
            "b.jpg": {
                "country": "COUNTRY_B",
            },
            "c.jpg": {
                "country": "COUNTRY_C",
            },
            "e.jpg": {
                "country": "COUNTRY_E",
            },
            "f.jpg": {
                "country": "COUNTRY_F",
            },
        }
        exif_results = {
            "a.jpg": blank_img.getexif(),
            "b.jpg": blank2_img.getexif(),
            "c.jpg": blank3_img.getexif(),
        }
        partial_predictions = {
            "f.jpg": {
                "filepath": "f.jpg",
                "classifications": {
                    "classes": ["XYZ"],
                    "scores": [0.8],
                },
                "detections": [],
                "prediction": "XYZ",
                "prediction_score": 0.4,
                "prediction_source": "partial",
                "model_version": expected_model_version,
            },
        }

        assert mock_ensemble2.combine(
            filepaths,
            classifier_results,
            detector_results,
            geolocation_results,
            exif_results,
            partial_predictions,
        ) == [
            {
                "filepath": "a.jpg",
                "failures": ["CLASSIFIER", "DETECTOR"],
                "country": "COUNTRY_A",
                "exif": {"DateTimeOriginal": "2016:09:25 18:35:10"},
                "model_version": expected_model_version,
            },
            {
                "filepath": "b.jpg",
                "failures": ["CLASSIFIER"],
                "country": "COUNTRY_B",
                "detections": [
                    {
                        "category": "1",
                        "label": "animal",
                        "conf": 0.5,
                        "bbox": [0.0, 0.1, 0.2, 0.3],
                    }
                ],
                "exif": {"DateTimeOriginal": "2016:07:31 15:16:34"},
                "model_version": expected_model_version,
            },
            {
                "filepath": "c.jpg",
                "failures": ["DETECTOR"],
                "country": "COUNTRY_C",
                "classifications": {
                    "classes": ["X", "Y", "Z"],
                    "scores": [0.5, 0.3, 0.2],
                },
                "exif": {"DateTimeOriginal": "2016:05:03 19:14:10"},
                "model_version": expected_model_version,
            },
            {
                "filepath": "d.jpg",
                "failures": ["GEOLOCATION"],
                "classifications": {
                    "classes": ["R", "S", "T"],
                    "scores": [0.7, 0.2, 0.1],
                },
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "prediction": "R",
                "prediction_score": 0.7,
                "prediction_source": "mock",
                "model_version": expected_model_version,
            },
            {
                "filepath": "e.jpg",
                "country": "COUNTRY_E",
                "classifications": {
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                },
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "prediction": "K",
                "prediction_score": 0.9,
                "prediction_source": "mock",
                "model_version": expected_model_version,
            },
            {
                "filepath": "f.jpg",
                "classifications": {
                    "classes": ["XYZ"],
                    "scores": [0.8],
                },
                "detections": [],
                "prediction": "XYZ",
                "prediction_score": 0.4,
                "prediction_source": "partial",
                "model_version": expected_model_version,
            },
        ]

    def test_complete_taxonomy(self, ensemble) -> None:

        missing_ancestors = set()

        taxonomy_levels = {
            "kingdom": 0,
            "class": 1,
            "order": 2,
            "family": 3,
            "genus": 4,
            "species": 5,
        }

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        def _level_idx_from_label(label: str) -> int:
            label_parts = label.split(";")
            for idx in range(5, 0, -1):
                if label_parts[idx]:
                    return idx
            return 0

        def _ancestor_from_label(label: str, level_idx: int) -> str:
            label_parts = label.split(";")
            return ";".join(label_parts[1 : level_idx + 1]) + (";" * (5 - level_idx))

        with open(
            ensemble.model_info.classifier_labels, mode="r", encoding="utf-8"
        ) as fp:
            classifier_labels = [line.strip() for line in fp.readlines()]

        for label in [Classification.HUMAN, Classification.ANIMAL]:
            taxa = _taxa_from_label(label)
            assert taxa in ensemble.taxonomy_map
            assert ensemble.taxonomy_map[taxa] == label
        for label in [Classification.UNKNOWN]:
            taxa = _taxa_from_label(label)
            assert taxa not in ensemble.taxonomy_map

        for label in classifier_labels:
            if label in [
                Classification.BLANK,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                continue

            max_level_idx = _level_idx_from_label(label)

            for taxonomy_level, taxonomy_level_idx in taxonomy_levels.items():
                ancestor = ensemble._get_ancestor_at_level(label, taxonomy_level)
                if taxonomy_level_idx <= max_level_idx:
                    if not ancestor:
                        missing_ancestors.add(
                            _ancestor_from_label(label, taxonomy_level_idx)
                        )
                else:
                    assert ancestor is None

        if missing_ancestors:
            warnings.warn(
                UserWarning(
                    "Missing from taxonomy: \n" + "\n".join(sorted(missing_ancestors))
                )
            )

    def test_complete_geofence(self, ensemble) -> None:

        missing_ancestors = set()

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        def _ancestor_from_label(label: str, level_idx: int) -> str:
            label_parts = label.split(";")
            return ";".join(label_parts[1 : level_idx + 1]) + (";" * (5 - level_idx))

        with open(
            ensemble.model_info.classifier_labels, mode="r", encoding="utf-8"
        ) as fp:
            classifier_labels = [line.strip() for line in fp.readlines()]

        for label in [
            Classification.BLANK,
            Classification.ANIMAL,
            Classification.VEHICLE,
            Classification.UNKNOWN,
        ]:
            taxa = _taxa_from_label(label)
            assert taxa not in ensemble.geofence_map

        for label in [Classification.HUMAN]:
            taxa = _taxa_from_label(label)
            assert taxa in ensemble.geofence_map

        for label in classifier_labels:
            if label in [
                Classification.BLANK,
                Classification.ANIMAL,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                continue

            for level_idx in range(1, 6):
                ancestor = _ancestor_from_label(label, level_idx)
                if ancestor not in ensemble.geofence_map:
                    missing_ancestors.add(ancestor)

        if missing_ancestors:
            warnings.warn(
                UserWarning(
                    "Missing from geofence: \n" + "\n".join(sorted(missing_ancestors))
                )
            )
