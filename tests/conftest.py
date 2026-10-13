from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from mockito import unstub

CONTRACT = yaml.safe_load((Path(__file__).parent.parent / "openapi.yaml").read_text())


@pytest.fixture(autouse=True)
def _unstub():
    yield
    unstub()


@pytest.fixture
def contract():
    def check(schema, payload):
        Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": CONTRACT["components"]}).validate(payload)

    return check
