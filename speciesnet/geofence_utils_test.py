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
# pylint: disable=redefined-outer-name

import pytest

from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
from speciesnet.geofence_utils import should_geofence_animal_classification

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


@pytest.fixture
def taxonomy_map():
    return {
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


@pytest.fixture
def geofence_map():
    return {
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


class TestGeofenceUtils:
    """Tests for the geofence utility functions."""

    def test_should_geofence_animal_classification(self, geofence_map) -> None:

        # Test when country is not provided.
        assert not should_geofence_animal_classification(
            LION,
            country=None,
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )

        # Test when label is not in the geofence map.
        assert not should_geofence_animal_classification(
            PUMA,
            country="USA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            PUMA,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=True,
        )

        # Test "allow" rules from the geofence map.
        assert should_geofence_animal_classification(
            LION,
            country="GBR",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            LION,
            country="KEN",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region="NY",
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=True,
        )

        # Test "block" rules from the geofence map.
        assert should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="FRA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="TZA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="USA",
            admin1_region="NY",
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert not should_geofence_animal_classification(
            SAND_CAT,
            country="GBR",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert should_geofence_animal_classification(
            SAND_CAT,
            country="AUS",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=True,
        )

    def test_should_geofence_animal_classification_disabled(self, geofence_map) -> None:

        # Test when country is not provided.
        assert not should_geofence_animal_classification(
            LION,
            country=None,
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )

        # Test when label is not in the geofence map.
        assert not should_geofence_animal_classification(
            PUMA,
            country="USA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            PUMA,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=False,
        )

        # Test "allow" rules from the geofence map.
        assert not should_geofence_animal_classification(
            LION,
            country="GBR",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            LION,
            country="KEN",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region="NY",
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            PANTHERA_GENUS,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=False,
        )

        # Test "block" rules from the geofence map.
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="FRA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="TZA",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="USA",
            admin1_region="CA",
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            FELIDAE_FAMILY,
            country="USA",
            admin1_region="NY",
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            SAND_CAT,
            country="GBR",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )
        assert not should_geofence_animal_classification(
            SAND_CAT,
            country="AUS",
            admin1_region=None,
            geofence_map=geofence_map,
            enable_geofence=False,
        )

    def test_roll_up_labels_to_first_matching_level(
        self, taxonomy_map, geofence_map
    ) -> None:
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
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["species"],
            non_blank_threshold=0.9,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]) is None
        assert rollup_fn([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]) is None
        assert rollup_fn([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == (
            BROWN_BEAR,
            pytest.approx(1.0),
            "classifier+rollup_to_species",
        )

        # Test rollups to genus level.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["genus"],
            non_blank_threshold=0.9,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.6, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01]) is None
        assert rollup_fn([0.7, 0.25, 0.01, 0.01, 0.01, 0.01, 0.01]) == (
            URSUS_GENUS,
            pytest.approx(0.95),
            "classifier+rollup_to_genus",
        )

        # Test rollups to family level.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["family"],
            non_blank_threshold=0.8,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.4, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]) is None
        assert rollup_fn([0.4, 0.21, 0.2, 0.0, 0.0, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.81),
            "classifier+rollup_to_family",
        )

        # Test rollups to order level.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["order"],
            non_blank_threshold=0.8,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.3, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0]) is None
        assert rollup_fn([0.3, 0.2, 0.1, 0.1, 0.23, 0.0, 0.0]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.83),
            "classifier+rollup_to_order",
        )

        # Test rollups to class level.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["class"],
            non_blank_threshold=0.8,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0]) is None
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.22, 0.1, 0.0]) == (
            MAMMALIA_CLASS,
            pytest.approx(0.82),
            "classifier+rollup_to_class",
        )

        # Test rollups to kingdom level.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["kingdom"],
            non_blank_threshold=0.81,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) is None
        assert rollup_fn([0.2, 0.2, 0.1, 0.1, 0.23, 0.1, 0.1]) == (
            ANIMAL_KINGDOM,
            pytest.approx(0.93),
            "classifier+rollup_to_kingdom",
        )

        # Test rollups when multiple taxonomy levels are specified.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["genus", "family", "order", "class", "kingdom"],
            non_blank_threshold=0.75,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.6, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.8),
            "classifier+rollup_to_family",
        )

        # Test rollups when multiple score sums pass the non blank threshold.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["species"],
            non_blank_threshold=0.1,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.2, 0.3, 0.15, 0.0, 0.35, 0.0, 0.0]) == (
            LION,
            pytest.approx(0.35),
            "classifier+rollup_to_species",
        )

        # Test rollups when the BLANK score dominates all the others.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country=None,
            admin1_region=None,
            target_taxonomy_levels=["species", "genus", "family", "order", "class"],
            non_blank_threshold=0.4,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.1, 0.2, 0.2, 0.45, 0.0, 0.0, 0.0]) == (
            URSIDAE_FAMILY,
            pytest.approx(0.5),
            "classifier+rollup_to_family",
        )

        # Test rollups with geofencing.
        rollup_fn = lambda scores: roll_up_labels_to_first_matching_level(
            labels=predictions,
            scores=scores,
            country="GBR",
            admin1_region=None,
            target_taxonomy_levels=["species", "genus", "family", "order", "class"],
            non_blank_threshold=0.4,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert rollup_fn([0.1, 0.2, 0.2, 0.45, 0.0, 0.0, 0.0]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.5),
            "classifier+rollup_to_order",
        )

        # Test rollups to invalid levels.
        with pytest.raises(ValueError):
            roll_up_labels_to_first_matching_level(
                labels=predictions,
                scores=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                country=None,
                admin1_region=None,
                target_taxonomy_levels=["invalid_level"],
                non_blank_threshold=0.3,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=True,
            )

    def test_geofence_animal_classification(self, taxonomy_map, geofence_map) -> None:
        # pylint: disable=unnecessary-lambda-assignment

        predictions = [
            LION,
            POLAR_BEAR,
            BLANK,
            FELIDAE_FAMILY,
        ]

        # Test when no geofencing is needed.
        geofence_fn = lambda scores: geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="TZA",
            admin1_region=None,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            LION,
            pytest.approx(0.4),
            "classifier",
        )

        # Test with geofencing and rollup to family level or above.
        geofence_fn = lambda scores: geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="USA",
            admin1_region=None,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            FELIDAE_FAMILY,
            pytest.approx(0.5),
            "classifier+geofence+rollup_to_family",
        )
        geofence_fn = lambda scores: geofence_animal_classification(
            labels=predictions,
            scores=scores,
            country="USA",
            admin1_region="NY",
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=True,
        )
        assert geofence_fn([0.4, 0.3, 0.2, 0.1]) == (
            CARNIVORA_ORDER,
            pytest.approx(0.8),
            "classifier+geofence+rollup_to_order",
        )
