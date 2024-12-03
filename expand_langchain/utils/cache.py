import json
import logging
from pathlib import Path

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from pydantic import BaseModel


def load_cache(path, key):
    path = Path(path)
    key = str(key).replace("/", "_")

    if (path / f"{key}.txt").exists():
        with open(path / f"{key}.txt", "r") as f:
            return f.read()
    elif (path / f"{key}.json").exists():
        with open(path / f"{key}.json", "r") as f:
            return json.load(f)
    elif (path / key).exists():
        # if path / key has integer keys, it is a list
        if all([p.stem.isdigit() for p in (path / key).iterdir()]):
            data = [None] * len(list((path / key).iterdir()))
        else:
            data = {}

        for p in (path / key).iterdir():
            if p.stem.isdigit():
                data[int(p.stem)] = load_cache(path / key, p.stem)
            else:
                data[p.stem] = load_cache(path / key, p.stem)

        return data
    else:
        raise FileNotFoundError(f"Cache not found: {path / key}")


def save_cache(path, key, data):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    key = str(key).replace("/", "_")

    if isinstance(data, str):
        path = path / f"{key}.txt"
        with open(path, "w") as f:
            f.write(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            save_cache(path / key, i, data[i])
    elif isinstance(data, dict):
        for k, v in data.items():
            save_cache(path / key, k, v)
    elif isinstance(data, Document):
        path = path / f"{key}.txt"
        with open(path, "w") as f:
            f.write(data.to_json())
    else:
        try:
            path = path / f"{key}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error in saving files: {e}")
