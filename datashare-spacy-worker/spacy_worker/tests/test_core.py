import json

import pytest
import requests
import spacy
from icij_common.test_utils import fail_if_exception
from packaging.version import Version

from spacy_worker.constants import DATA_DIR
from spacy_worker.core import SpacyProvider
from spacy_worker.objects import SpacySize

_MODEL_PATH = DATA_DIR.joinpath("models.json")
_ALL_LANGUAGES = list(json.loads(_MODEL_PATH.read_text()))


@pytest.fixture(scope="session")
def test_spacy_provider() -> SpacyProvider:
    return SpacyProvider(max_languages=1)


def test_spacy_provider_get_ner(test_spacy_provider: SpacyProvider):
    # Given
    language = "en"
    provider = test_spacy_provider
    # When/Then
    msg = f"failed to load ner pipeline for {language}"
    with fail_if_exception(msg):
        provider.get_ner(language, size=SpacySize.SMALL)


def test_spacy_provider_get_sent_split(test_spacy_provider: SpacyProvider):
    # Given
    language = "en"
    provider = test_spacy_provider
    # When/Then
    msg = f"failed to load sentence split pipeline for {language}"
    with fail_if_exception(msg):
        provider.get_sent_split(language)


@pytest.mark.skip
def test_spacy_model_compatibility():
    # Given
    compatibility_url = (
        "https://raw.githubusercontent.com/explosion/spacy-models/"
        "master/compatibility.json"
    )
    r = requests.get(compatibility_url)
    r.raise_for_status()
    version = Version(spacy.__version__)
    version = f"{version.major}.{version.minor}"
    version_compatibility = r.json()["spacy"][version]
    models = json.loads(_MODEL_PATH.read_text())

    # Then
    for language, model in models.items():
        compatible_versions = version_compatibility[model["model"]]
        assert model["version"] in compatible_versions
