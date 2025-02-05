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

"""Taxonomy-related utility functions.

Provides functions for working with taxonomic labels, such as finding ancestors at
specific levels and extracting full class strings.
"""

from typing import Optional

from speciesnet.constants import Classification


def get_ancestor_at_level(
    label: str, taxonomy_level: str, taxonomy_map: dict
) -> Optional[str]:
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
        taxonomy_map:
            Dictionary mapping taxa to labels.

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
    return taxonomy_map.get(ancestor)


def get_full_class_string(label: str) -> str:
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
