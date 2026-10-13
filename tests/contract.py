from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

CONTRACTS = Path(__file__).parent.parent / "contracts"


def contract_schemas():
    schemas = {}
    for path in sorted(CONTRACTS.glob("*.yaml")):
        schemas.update(yaml.safe_load(path.read_text())["components"]["schemas"])
    return schemas


def contract(schema, payload):
    schemas = contract_schemas()
    if schema not in schemas:
        pytest.skip(f"no contract in contracts/ defines {schema}")
    Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": {"schemas": schemas}}).validate(payload)
