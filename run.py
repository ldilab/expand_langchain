import fire

from expand_langchain.config import Config
from expand_langchain.generator import Generator
from expand_langchain.loader import Loader

if __name__ == "__main__":
    fire.Fire(
        {
            "config": Config,
            "loader": Loader,
            "generator": Generator,
        }
    )
