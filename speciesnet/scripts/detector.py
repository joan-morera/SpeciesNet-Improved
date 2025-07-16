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

"""Detector functionality of SpeciesNet."""

__all__ = [
    "SpeciesNetDetector",
]

import time
from typing import Any, Optional

from absl import logging
from humanfriendly import format_timespan
import numpy as np
import PIL.Image
import torch
import torch.backends
import torch.backends.mps
from ultralytics import YOLO

from speciesnet.constants import Detection
from speciesnet.constants import Failure
from speciesnet.utils import ModelInfo
from speciesnet.utils import PreprocessedImage


class SpeciesNetDetector:
    """Detector component of SpeciesNet."""

    IMG_SIZE = 1280
    STRIDE = 64
    DETECTION_THRESHOLD = 0.01

    def __init__(self, model_name: str, yolov10_model_name: Optional[str] = None) -> None:
        """Loads the detector resources.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
            yolov10_model_name:
                Optional string value identifying a specific YOLOv10 model to use from
                `speciesnet.constants.YOLOV10_MODELS`. If not provided, the "default"
                YOLOv10 model will be used.
        """

        start_time = time.time()
        logging.info(f"SpeciesNetDetector __init__: model_name={model_name}, yolov10_model_name={yolov10_model_name}")

        self.model_info = ModelInfo(model_name, yolov10_model_name=yolov10_model_name)

        """
        To use a different YOLOv10 model from the PyTorch Wildlife project
        (https://microsoft.github.io/CameraTraps/model_zoo/megadetector/),
        update the `YOLOV10_MODEL_URL` constant in `speciesnet/constants.py`.
        """
        # Select the best device available.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logging.info(f"SpeciesNetDetector selected device: {self.device}")

        # Load the model.
        try:
            self.model = YOLO(self.model_info.detector)
            self.model.to(self.device)
            logging.info(f"YOLO model loaded successfully: {self.model_info.detector}")
        except Exception as e:
            logging.error(f"Error loading YOLO model {self.model_info.detector}: {e}", exc_info=True)
            raise

        end_time = time.time()
        logging.info(f"SpeciesNetDetector initialization took {format_timespan(end_time - start_time)}.")


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

        return PreprocessedImage(np.asarray(img), img.width, img.height)

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
                "failures": [Failure.DETECTOR.name],
            }

        # Run inference.
        try:
            results = self.model.predict(
                source=img.arr,
                conf=self.DETECTION_THRESHOLD,
                imgsz=self.IMG_SIZE,
                device=self.device,
                verbose=False,
            )[0]
            logging.info(f"YOLO model predict results: {results}")
        except Exception as e:
            logging.error(f"Error during YOLO model prediction for {filepath}: {e}", exc_info=True)
            return {
                "filepath": filepath,
                "failures": [Failure.DETECTOR.name, f"PredictionError: {e}"],
            }

        if self.device == "mps":
            results = results.cpu()

        # Process detections.
        detections = []
        boxes = results.boxes
        for i in range(len(boxes)):
            xywhn = boxes.xywhn[i]
            bbox = self._convert_yolo_xywhn_to_md_xywhn(xywhn.tolist())

            conf = boxes.conf[i].item()

            category = str(int(boxes.cls[i].item()) + 1)
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
