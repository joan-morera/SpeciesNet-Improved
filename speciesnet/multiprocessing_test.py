import json
import logging
import multiprocessing as mp

import pytest

from speciesnet.multiprocessing import SpeciesNet


@pytest.fixture(scope="session", autouse=True)
def always_mp_spawn():
    mp.set_start_method("spawn")


@pytest.fixture(name="instances_dict")
def fx_instances_dict() -> dict:
    with open("test_data/instances_with_errors.json", mode="r", encoding="utf-8") as fp:
        return json.load(fp)


class TestSingleProcess:

    @pytest.fixture(scope="class")
    def model(self, model_name: str) -> SpeciesNet:
        return SpeciesNet(model_name)

    def test_predict(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.predict(
            instances_dict=instances_dict, run_mode="single_thread", progress_bars=True
        )
        predictions_dict2 = model.predict(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict1 == predictions_dict2
        logging.info("Predictions (%s): %s", request.node.name, predictions_dict1)

    def test_classify(self, request, instances_dict, model) -> None:
        predictions_dict = model.classify(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict
        logging.info("Classifications (%s): %s", request.node.name, predictions_dict)

    def test_detect(self, request, instances_dict, model) -> None:
        predictions_dict = model.detect(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict
        logging.info("Detections (%s): %s", request.node.name, predictions_dict)


class TestMultiProcess:

    @pytest.fixture(scope="class")
    def model(self, model_name: str) -> SpeciesNet:
        return SpeciesNet(model_name, multiprocessing=True)

    def test_predict(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.predict(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.predict(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict1 == predictions_dict2
        logging.info("Predictions (%s): %s", request.node.name, predictions_dict1)

    def test_classify(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.classify(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.classify(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict1 == predictions_dict2
        logging.info("Classifications (%s): %s", request.node.name, predictions_dict1)

    def test_detect(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.detect(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.detect(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict1 == predictions_dict2
        logging.info("Detections (%s): %s", request.node.name, predictions_dict1)
