import json
import logging
from datetime import date
from pathlib import Path

from langchain_core.documents import Document


def load_cache(path, key):
    path = Path(path)
    key = str(key).replace("/", "_")

    if (path / f"{key}.pkl").exists():
        with open(path / f"{key}.pkl", "rb") as f:
            import pickle

            return pickle.load(f)
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

    path = path / f"{key}.pkl"
    with open(path, "wb") as f:
        import pickle

        pickle.dump(data, f)

    logging.info(f"Saved cache to {path}, key: {key}")
