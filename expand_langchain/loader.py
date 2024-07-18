import json
import logging
import os
from pathlib import Path

import psycopg2
import yaml
from datasets import Dataset, load_dataset
from expand_langchain.config import Config, DatasetConfig, SourceConfig
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Loader(BaseModel):
    config: Config = None
    path: str = None

    result: dict = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.path is not None:
            self.config = Config(path=self.path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def run(self):
        sources = self.load_sources()
        self.result = self.load_datasets(sources)

        return self

    def load_sources(self):
        sources = {}
        for source in self.config.source:
            source: SourceConfig
            name = source.name

            if source.type == "huggingface":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                split = source.kwargs.get("split")
                sources[name] = load_dataset(path)[split].sort(sort_key)

            elif source.type == "json":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_json = json.loads(Path(path).read_text())
                sources[name] = Dataset.from_list(data_json).sort(sort_key)

            elif source.type == "yaml":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_yaml = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)
                sources[name] = Dataset.from_list(data_yaml).sort(sort_key)

            elif source.type == "postgresql":
                sources[name] = psycopg2.connect(
                    dbname=source.kwargs.get("dbname"),
                    user=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                    host=os.getenv("POSTGRES_HOST"),
                    port=os.getenv("POSTGRES_PORT"),
                )
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

        return sources

    def load_datasets(self, sources):
        datasets = {}
        for dataset in self.config.dataset:
            dataset: DatasetConfig

            name = dataset.name
            if dataset.type == "dict":
                datasets[name] = _load_dict(sources, **dataset.kwargs)

            elif dataset.type == "schema":
                datasets[name] = _load_schema(sources, **dataset.kwargs)

            else:
                raise ValueError(f"Unknown dataset type: {dataset.type}")

        return datasets


def _load_dict(sources, primary_key, fields):
    result = {}

    primary_field = list(filter(lambda x: x.get("name") == primary_key, fields))[0]
    ids = sources[primary_field.get("source")][primary_field.get("key")]
    for i, id in tqdm(enumerate(ids)):
        result[id] = {}
        for field in fields:
            source = sources[field.get("source")]
            result[id][field.get("name")] = source[i][field.get("key")]

    return result


def _load_schema(sources_dict, sources):
    results = []
    for key in sources:
        conn = sources_dict[key]
        table_names = []
        table_descriptions = []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    n.nspname AS schema,
                    c.relname AS table,
                    pg_catalog.obj_description(c.oid, 'pg_class') AS description
                FROM
                    pg_catalog.pg_class c
                JOIN
                    pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE
                    c.relkind = 'r' -- only tables
                    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY
                    n.nspname, c.relname;
                """
            )
            for schema, table, description in cur.fetchall():
                table_names.append(f"{schema}.{table}")
                table_descriptions.append(description)

        tables = []
        for table, table_description in zip(table_names, table_descriptions):
            schema, table = table.split(".")
            columns = []
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        n.nspname AS schema,
                        c.relname AS table,
                        a.attname AS column,
                        CASE
                            WHEN t.typname = 'numeric' THEN
                                'numeric(' || ((a.atttypmod - 4) >> 16) || ',' || ((a.atttypmod - 4) & 65535) || ')'
                            WHEN t.typname = 'varchar' THEN
                                'varchar(' || (a.atttypmod - 4) || ')'
                            WHEN t.typname = 'char' THEN
                                'char(' || (a.atttypmod - 4) || ')'
                            ELSE
                                t.typname
                        END AS data_type,
                        pg_catalog.col_description(a.attrelid, a.attnum) AS description
                    FROM
                        pg_catalog.pg_class c
                    JOIN
                        pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    JOIN
                        pg_catalog.pg_attribute a ON a.attrelid = c.oid
                    JOIN
                        pg_catalog.pg_type t ON t.oid = a.atttypid
                    WHERE
                        c.relkind = 'r' -- only tables
                        AND a.attnum > 0
                        AND NOT a.attisdropped
                        AND n.nspname = %s
                        AND c.relname = %s
                    ORDER BY
                        a.attnum;
                    """,
                    (schema, table),
                )
                for schema, table, column, data_type, description in cur.fetchall():
                    columns.append(
                        {
                            "name": column,
                            "description": description,
                            "type": data_type,
                        }
                    )

            tables.append(
                {
                    "name": table,
                    "description": table_description,
                    "columns": columns,
                }
            )

        result = {key: tables}
        results.append(result)

    return results
