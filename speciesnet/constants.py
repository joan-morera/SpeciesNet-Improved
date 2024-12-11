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

"""Useful constants."""

__all__ = [
    "Classification",
    "Detection",
]

import enum
from typing import Optional


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
