import json
import logging
from datetime import date
from pathlib import Path

from langchain_core.documents import Document


def load_cache(path, key):
    path = Path(path)
    key = str(key).replace("/", "_")

    if (path / f"{key}.txt").exists():
        with open(path / f"{key}.txt", "r") as f:
            return f.read()
    elif (path / f"{key}.json").exists():
        with open(path / f"{key}.json", "r") as f:
            return json.load(f)
    elif (path / f"{key}.pkl").exists():
        with open(path / f"{key}.pkl", "rb") as f:
            import pickle

            return pickle.load(f)
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


def remove_cache(path, key):
    path = Path(path)
    key = str(key).replace("/", "_")

    if (path / f"{key}.txt").exists():
        (path / f"{key}.txt").unlink()
    elif (path / f"{key}.json").exists():
        (path / f"{key}.json").unlink()
    elif (path / f"{key}.pkl").exists():
        (path / f"{key}.pkl").unlink()
    elif (path / key).exists():
        for p in (path / key).iterdir():
            remove_cache(path / key, p.stem)
            remove_cache(path / key, p.name)
        (path / key).rmdir()


def save_cache(path, key, data):
    path = Path(path)
    key = str(key).replace("/", "_")

    remove_cache(path, key)
    path.mkdir(parents=True, exist_ok=True)

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
    elif isinstance(data, date):
        path = path / f"{key}.txt"
        with open(path, "w") as f:
            f.write(data.isoformat())
    else:
        path = path / f"{key}.pkl"
        with open(path, "wb") as f:
            import pickle

            pickle.dump(data, f)

    logging.info(f"Saved cache to {path}, key: {key}")
