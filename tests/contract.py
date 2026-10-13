from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

CONTRACT = yaml.safe_load((Path(__file__).parent.parent / "openapi.yaml").read_text())


def contract(schema, payload):
    Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": CONTRACT["components"]}).validate(payload)
