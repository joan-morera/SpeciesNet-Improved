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

"""Script to run the SpeciesNet model.

Provides a command-line interface to execute the SpeciesNet model on various
inputs. It uses flags for specifying input, output, and run options, allowing 
the user to run the model in different modes.
"""

import json
import multiprocessing as mp

from absl import app
from absl import flags
from absl import logging

from speciesnet import DEFAULT_MODEL
from speciesnet import only_one_true
from speciesnet import SpeciesNet

_MODEL = flags.DEFINE_string(
    "model",
    DEFAULT_MODEL,
    "SpeciesNet model to load.",
)
_GEOFENCE = flags.DEFINE_bool(
    "geofence",
    True,
    "Whether to enable geofencing or not.",
)
_INSTANCES_JSON = flags.DEFINE_string(
    "instances_json",
    None,
    "Input JSON file with instances to get predictions for.",
)
_FILEPATHS = flags.DEFINE_list(
    "filepaths",
    None,
    "List of image filepaths to get predictions for.",
)
_FILEPATHS_TXT = flags.DEFINE_string(
    "filepaths_txt",
    None,
    "Input TXT file with image filepaths to get predictions for.",
)
_FOLDERS = flags.DEFINE_list(
    "folders",
    None,
    "List of image folders to get predictions for.",
)
_FOLDERS_TXT = flags.DEFINE_string(
    "folders_txt",
    None,
    "Input TXT file with image folders to get predictions for.",
)
_PREDICTIONS_JSON = flags.DEFINE_string(
    "predictions_json",
    None,
    "Output JSON file for storing computed predictions.",
)
_RUN_MODE = flags.DEFINE_enum(
    "run_mode",
    "multi_thread",
    ["single_thread", "multi_thread", "multi_process"],
    "Running mode, determining the parallelism strategy to use at prediction time.",
)
_PROGRESS_BARS = flags.DEFINE_bool(
    "progress_bars",
    True,
    "Whether to show progress bars for the various inference components.",
)


def main(argv: list[str]) -> None:
    del argv  # Unused.

    # Check for valid inputs.
    inputs = [_INSTANCES_JSON, _FILEPATHS, _FILEPATHS_TXT, _FOLDERS, _FOLDERS_TXT]
    inputs_names = [i.name for i in inputs]
    inputs_values = [i.value for i in inputs]
    if not only_one_true(*inputs_values):
        raise ValueError(
            "Expected exactly one of ["
            f"{', '.join([f'--{name}' for name in inputs_names])}"
            f"] to be provided. Received: {inputs_values}."
        )

    # Ask the user to confirm that they want to continue without writing the predictions
    # to a JSON file.
    if not _PREDICTIONS_JSON.value:
        user_input = input(
            "Continue without saving predictions to a JSON file? [y/N]: "
        )
        if user_input.lower() not in ["yes", "y"]:
            print("Please provide an output filepath via --predictions_json.")
            return

    # Set running mode.
    run_mode = _RUN_MODE.value
    mp.set_start_method("spawn")

    # Make predictions.
    model = SpeciesNet(
        _MODEL.value,
        geofence=_GEOFENCE.value,
        multiprocessing=(run_mode == "multi_process"),
    )
    predictions_dict = model.predict(
        instances_json=_INSTANCES_JSON.value,
        filepaths=_FILEPATHS.value,
        filepaths_txt=_FILEPATHS_TXT.value,
        folders=_FOLDERS.value,
        folders_txt=_FOLDERS_TXT.value,
        run_mode=run_mode,
        progress_bars=_PROGRESS_BARS.value,
        predictions_json=_PREDICTIONS_JSON.value,
    )
    if predictions_dict is not None:
        logging.info(
            "Predictions:\n%s",
            json.dumps(predictions_dict, ensure_ascii=False, indent=4),
        )


if __name__ == "__main__":
    app.run(main)
