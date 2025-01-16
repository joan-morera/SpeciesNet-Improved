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

from speciesnet.taxonomy_utils import get_ancestor_at_level
from speciesnet.taxonomy_utils import get_full_class_string

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


class TestTaxonomyUtils:
    """Tests for the taxonomy utility functions."""

    def test_get_ancestor_at_level(self, taxonomy_map) -> None:

        # Test all ancestors of LION.
        assert get_ancestor_at_level(LION, "species", taxonomy_map) == LION
        assert get_ancestor_at_level(LION, "genus", taxonomy_map) == PANTHERA_GENUS
        assert get_ancestor_at_level(LION, "family", taxonomy_map) == FELIDAE_FAMILY
        assert get_ancestor_at_level(LION, "order", taxonomy_map) == CARNIVORA_ORDER
        assert get_ancestor_at_level(LION, "class", taxonomy_map) == MAMMALIA_CLASS
        assert get_ancestor_at_level(LION, "kingdom", taxonomy_map) == ANIMAL_KINGDOM

        # Test all ancestors of PANTHERA_GENUS.
        assert get_ancestor_at_level(PANTHERA_GENUS, "species", taxonomy_map) is None
        assert (
            get_ancestor_at_level(PANTHERA_GENUS, "genus", taxonomy_map)
            == PANTHERA_GENUS
        )
        assert (
            get_ancestor_at_level(PANTHERA_GENUS, "family", taxonomy_map)
            == FELIDAE_FAMILY
        )
        assert (
            get_ancestor_at_level(PANTHERA_GENUS, "order", taxonomy_map)
            == CARNIVORA_ORDER
        )
        assert (
            get_ancestor_at_level(PANTHERA_GENUS, "class", taxonomy_map)
            == MAMMALIA_CLASS
        )
        assert (
            get_ancestor_at_level(PANTHERA_GENUS, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of FELIDAE_FAMILY.
        assert get_ancestor_at_level(FELIDAE_FAMILY, "species", taxonomy_map) is None
        assert get_ancestor_at_level(FELIDAE_FAMILY, "genus", taxonomy_map) is None
        assert (
            get_ancestor_at_level(FELIDAE_FAMILY, "family", taxonomy_map)
            == FELIDAE_FAMILY
        )
        assert (
            get_ancestor_at_level(FELIDAE_FAMILY, "order", taxonomy_map)
            == CARNIVORA_ORDER
        )
        assert (
            get_ancestor_at_level(FELIDAE_FAMILY, "class", taxonomy_map)
            == MAMMALIA_CLASS
        )
        assert (
            get_ancestor_at_level(FELIDAE_FAMILY, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of CARNIVORA_ORDER.
        assert get_ancestor_at_level(CARNIVORA_ORDER, "species", taxonomy_map) is None
        assert get_ancestor_at_level(CARNIVORA_ORDER, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(CARNIVORA_ORDER, "family", taxonomy_map) is None
        assert (
            get_ancestor_at_level(CARNIVORA_ORDER, "order", taxonomy_map)
            == CARNIVORA_ORDER
        )
        assert (
            get_ancestor_at_level(CARNIVORA_ORDER, "class", taxonomy_map)
            == MAMMALIA_CLASS
        )
        assert (
            get_ancestor_at_level(CARNIVORA_ORDER, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of MAMMALIA_CLASS.
        assert get_ancestor_at_level(MAMMALIA_CLASS, "species", taxonomy_map) is None
        assert get_ancestor_at_level(MAMMALIA_CLASS, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(MAMMALIA_CLASS, "family", taxonomy_map) is None
        assert get_ancestor_at_level(MAMMALIA_CLASS, "order", taxonomy_map) is None
        assert (
            get_ancestor_at_level(MAMMALIA_CLASS, "class", taxonomy_map)
            == MAMMALIA_CLASS
        )
        assert (
            get_ancestor_at_level(MAMMALIA_CLASS, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of ANIMAL_KINGDOM.
        assert get_ancestor_at_level(ANIMAL_KINGDOM, "species", taxonomy_map) is None
        assert get_ancestor_at_level(ANIMAL_KINGDOM, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(ANIMAL_KINGDOM, "family", taxonomy_map) is None
        assert get_ancestor_at_level(ANIMAL_KINGDOM, "order", taxonomy_map) is None
        assert get_ancestor_at_level(ANIMAL_KINGDOM, "class", taxonomy_map) is None
        assert (
            get_ancestor_at_level(ANIMAL_KINGDOM, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test all ancestors of BLANK.
        assert get_ancestor_at_level(BLANK, "species", taxonomy_map) is None
        assert get_ancestor_at_level(BLANK, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(BLANK, "family", taxonomy_map) is None
        assert get_ancestor_at_level(BLANK, "order", taxonomy_map) is None
        assert get_ancestor_at_level(BLANK, "class", taxonomy_map) is None
        assert get_ancestor_at_level(BLANK, "kingdom", taxonomy_map) is None

        # Test all ancestors of HUMAN, when its genus, family and order are missing from
        # the mock taxonomy mapping.
        assert get_ancestor_at_level(HUMAN, "species", taxonomy_map) == HUMAN
        assert get_ancestor_at_level(HUMAN, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(HUMAN, "family", taxonomy_map) is None
        assert get_ancestor_at_level(HUMAN, "order", taxonomy_map) is None
        assert get_ancestor_at_level(HUMAN, "class", taxonomy_map) == MAMMALIA_CLASS
        assert get_ancestor_at_level(HUMAN, "kingdom", taxonomy_map) == ANIMAL_KINGDOM

        # Test all ancestors of VEHICLE.
        assert get_ancestor_at_level(VEHICLE, "species", taxonomy_map) is None
        assert get_ancestor_at_level(VEHICLE, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(VEHICLE, "family", taxonomy_map) is None
        assert get_ancestor_at_level(VEHICLE, "order", taxonomy_map) is None
        assert get_ancestor_at_level(VEHICLE, "class", taxonomy_map) is None
        assert get_ancestor_at_level(VEHICLE, "kingdom", taxonomy_map) is None

        # Test all ancestors of an unseen species.
        unseen_species = "uuid;class;order;family;genus;species;common_name"
        assert get_ancestor_at_level(unseen_species, "species", taxonomy_map) is None
        assert get_ancestor_at_level(unseen_species, "genus", taxonomy_map) is None
        assert get_ancestor_at_level(unseen_species, "family", taxonomy_map) is None
        assert get_ancestor_at_level(unseen_species, "order", taxonomy_map) is None
        assert get_ancestor_at_level(unseen_species, "class", taxonomy_map) is None
        assert (
            get_ancestor_at_level(unseen_species, "kingdom", taxonomy_map)
            == ANIMAL_KINGDOM
        )

        # Test errors due to invalid labels.
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species"
            get_ancestor_at_level(invalid_label, "kingdom", taxonomy_map)
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species;common_name;extra"
            get_ancestor_at_level(invalid_label, "kingdom", taxonomy_map)

    def test_get_full_class_string(self) -> None:

        # Test BLANK/HUMAN/VEHICLE.
        assert get_full_class_string(BLANK) == BLANK_FC
        assert get_full_class_string(HUMAN) == HUMAN_FC
        assert get_full_class_string(VEHICLE) == VEHICLE_FC

        # Test valid labels at different taxonomy levels.
        assert get_full_class_string(LION) == LION_FC
        assert get_full_class_string(PANTHERA_GENUS) == PANTHERA_GENUS_FC
        assert get_full_class_string(FELIDAE_FAMILY) == FELIDAE_FAMILY_FC
        assert get_full_class_string(CARNIVORA_ORDER) == CARNIVORA_ORDER_FC
        assert get_full_class_string(MAMMALIA_CLASS) == MAMMALIA_CLASS_FC
        assert get_full_class_string(ANIMAL_KINGDOM) == ANIMAL_KINGDOM_FC

        # Test errors due to invalid labels.
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species"
            get_full_class_string(invalid_label)
        with pytest.raises(ValueError):
            invalid_label = "uuid;class;order;family;genus;species;common_name;extra"
            get_full_class_string(invalid_label)
