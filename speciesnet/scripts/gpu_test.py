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

"""Script to check GPU availability for both TensorFlow and PyTorch in the current
Python environment.
"""

import os

# Minimize TF printouts
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

try:
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
except Exception:
    pass


def torch_test():
    """Print diagnostic information about Torch/CUDA status, including Torch/CUDA
    versions and all available CUDA device names.
    """

    try:
        import torch
    except Exception as e:
        print(
            "PyTorch unavailable, not running PyTorch tests. "
            "PyTorch import error was:\n{}".format(str(e))
        )
        return

    print("Torch version: {}".format(str(torch.__version__)))
    print("CUDA available (according to PyTorch): {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA version (according to PyTorch): {}".format(torch.version.cuda))  # type: ignore
        print(
            "CuDNN version (according to PyTorch): {}".format(
                torch.backends.cudnn.version()
            )
        )

    device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) > 0:
        cuda_str = "Found {} CUDA devices:".format(len(device_ids))
        print(cuda_str)

        for device_id in device_ids:
            device_name = "unknown"
            try:
                device_name = torch.cuda.get_device_name(device=device_id)
            except Exception:
                pass
            print("{}: {}".format(device_id, device_name))
    else:
        print("No GPUs reported by PyTorch")

    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            print("PyTorch reports that Metal Performance Shaders are available")
    except Exception:
        pass
    return len(device_ids)


def tf_test():
    """Print diagnostic information about TF/CUDA status."""

    try:
        import tensorflow as tf
    except Exception as e:
        print(
            "TensorFlow unavailable, not running TF tests. "
            "TF import error was:\n{}".format(str(e))
        )
        return

    from tensorflow.python.platform import build_info as build

    print(f"TF version: {tf.__version__}")

    if "cuda_version" not in build.build_info:
        print("TF does not appear to be built with CUDA")
    else:
        print(
            "CUDA build version reported by TensorFlow:",
            build.build_info["cuda_version"],
        )
    if "cudnn_version" not in build.build_info:
        print("TF does not appear to be built with CuDNN")
    else:
        print(
            "CuDNN build version reported by TensorFlow:",
            build.build_info["cudnn_version"],
        )

    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        print(
            "Linked TensorRT version: {}".format(
                trt.trt_utils._pywrap_py_utils.get_linked_tensorrt_version()
            )
        )
    except Exception:
        print("Could not probe TensorRT version")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus is None:
        gpus = []

    if len(gpus) > 0:
        print("TensorFlow found the following GPUs:")
        for gpu in gpus:
            print(gpu.name)

    else:
        print("No GPUs reported by TensorFlow")

    return len(gpus)


if __name__ == "__main__":

    print("*** Running Torch tests ***\n")
    torch_test()

    print("\n*** Running TF tests ***\n")
    tf_test()
