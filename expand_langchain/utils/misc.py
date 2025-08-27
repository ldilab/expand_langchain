from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import PlainScalarString


def pretty_yaml_dump(data, path):
    def _long_string_representer(dumper, data):
        data = data.replace("\r", "")
        data = PlainScalarString(data)

        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    def _default_representer(dumper, data):
        data = str(data)
        return _long_string_representer(dumper, data)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.representer.add_representer(object, _default_representer)
    yaml.representer.add_representer(str, _long_string_representer)

    with open(path, "w") as f:
        yaml.dump(data, f)
