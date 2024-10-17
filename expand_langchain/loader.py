import json
import logging
import os
import pprint
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from elasticsearch import Elasticsearch, helpers
from pydantic import BaseModel
from tqdm import tqdm

from expand_langchain.config import Config, DatasetConfig, SourceConfig

logger = logging.getLogger(__name__)


class Loader(BaseModel):
    config: Config = None
    config_path: str = None
    api_keys_path: str = "api_keys.json"

    result: Any = None

    def __init__(self, **data):
        super().__init__(**data)

        self._load_config()
        self._load_api_keys()

    def _load_config(self):
        if self.config_path is not None:
            self.config = Config(path=self.config_path)
        elif self.config is None:
            raise ValueError("Either config_path or config should be provided")
        else:
            pass

    def _load_api_keys(self):
        api_keys = json.loads(Path(self.api_keys_path).read_text())
        for k, v in api_keys.items():
            os.environ[k] = v

    def run(self):
        sources = self.load_sources()
        self.result = self._load_datasets(sources)

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
                load_dataset_kwargs = source.kwargs.get("load_dataset_kwargs", {})
                dataset = load_dataset(path, **load_dataset_kwargs)[split]
                sources[name] = dataset.sort(sort_key)

            elif source.type == "json":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_json = json.loads(Path(path).read_text())
                sources[name] = Dataset.from_list(data_json).sort(sort_key)

            elif source.type == "jsonl":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_jsonl = [
                    json.loads(line)
                    for line in Path(path).read_text().split("\n")
                    if line
                ]
                sources[name] = Dataset.from_list(data_jsonl).sort(sort_key)

            elif source.type == "yaml":
                path = source.kwargs.get("path")
                sort_key = source.kwargs.get("sort_key")
                data_yaml = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)
                sources[name] = Dataset.from_list(data_yaml).sort(sort_key)

            elif source.type == "postgresql":
                import psycopg2

                sources[name] = psycopg2.connect(
                    dbname=source.kwargs.get("dbname"),
                    user=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                    host=os.getenv("POSTGRES_HOST"),
                    port=os.getenv("POSTGRES_PORT"),
                )

            elif source.type == "user_input":
                sources[name] = None
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

        return sources

    def _load_datasets(self, sources):
        datasets = {}
        for dataset in self.config.dataset:
            dataset: DatasetConfig

            name = dataset.name
            if dataset.type == "dict":
                result = _load_dict(sources, **dataset.kwargs)

            elif dataset.type == "db-schema":
                if dataset.remove and not dataset.kwargs.get("rerun"):
                    continue
                result = _load_db_schema(sources, **dataset.kwargs)

            elif dataset.type == "db-value":
                if dataset.remove and not dataset.kwargs.get("rerun"):
                    continue
                result = _load_db_value(sources, **dataset.kwargs)

            elif dataset.type == "user_input":
                result = None

            else:
                raise ValueError(f"Unknown dataset type: {dataset.type}")

            if not dataset.remove:
                datasets[name] = result

        return datasets

    def save(self, path):
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)

        with open(path, "w") as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)

        return self

    def exit(self):
        pass


def _load_dict(
    sources,
    primary_key,
    fields,
    query: str = None,
    cache_dir: str = None,
    custom_lambda: str = None,
):
    config = {
        "primary_key": primary_key,
        "fields": fields,
        "query": query,
    }
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        config_path = cache_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                cache_config = json.load(f)
            if cache_config == config:
                data_path = cache_dir / "data.json"
                if data_path.exists():
                    with open(data_path, "r") as f:
                        return json.load(f)

    result = {}

    primary_field = list(filter(lambda x: x.get("name") == primary_key, fields))[0]
    ids = sources[primary_field.get("source")][primary_field.get("key")]
    for i, id in tqdm(enumerate(ids)):
        result[id] = {}
        for field in fields:
            source = sources[field.get("source")]
            result[id][field.get("name")] = source[i][field.get("key")]

    if custom_lambda is not None:
        try:
            func_obj = eval(custom_lambda)
        except:
            local_namespace = {}
            exec(custom_lambda, globals(), local_namespace)
            func_obj = local_namespace["func"]

        result = {k: func_obj(v) for k, v in result.items()}

    if query is not None:
        from tinydb import TinyDB, where
        from tinydb.storages import MemoryStorage

        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple(list(result.values()))
        result = db.search(eval(query, {"where": where}))
        result = {r[primary_key]: r for r in result}

    if cache_dir is not None:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        with open(cache_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        with open(cache_dir / "data.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _load_db_schema(sources_dict, sources, index_name, rerun=False):
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

    if rerun:
        client = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL"),
            api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        )

        index_name = index_name.lower()
        client.indices.delete(index=index_name, ignore=[400, 404])
        client.indices.create(index=index_name)

        for result in results:
            for db_id, tables in result.items():

                def generate_docs():
                    for table in tables:
                        yield {
                            "_index": index_name,
                            "_source": {
                                "db_id": db_id,
                                "table": table.get("name"),
                                "table_description": table.get("description"),
                                "column": "*",
                                "column_description": "",
                                "type": "",
                            },
                        }
                        for column in table.get("columns"):
                            yield {
                                "_index": index_name,
                                "_source": {
                                    "db_id": db_id,
                                    "table": table.get("name"),
                                    "table_description": table.get("description"),
                                    "column": column.get("name"),
                                    "column_description": column.get("description"),
                                    "type": column.get("type"),
                                },
                            }

                helpers.bulk(client, generate_docs())

    return results


def _load_db_value(sources_dict, sources, index_name, rerun=False):
    results = []
    for key in sources:
        conn = sources_dict[key]
        table_names = []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    n.nspname AS schema,
                    c.relname AS table
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
            for schema, table in cur.fetchall():
                table_names.append(f"{schema}.{table}")

        data_tuples = []
        for table_name in table_names:
            schema, table = table_name.split(".")
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        n.nspname AS schema,
                        c.relname AS table,
                        a.attname AS column
                    FROM
                        pg_catalog.pg_class c
                    JOIN
                        pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    JOIN
                        pg_catalog.pg_attribute a ON a.attrelid = c.oid
                    WHERE
                        c.relkind = 'r' -- only tables
                        AND a.attnum > 0
                        AND NOT a.attisdropped
                        AND n.nspname = %s
                        AND c.relname = %s
                    ORDER BY
                        a.attnum;""",
                    (schema, table),
                )

                columns = cur.fetchall()

                for schema_name, table_name, column_name in columns:
                    cur.execute(
                        f'SELECT "{column_name}" FROM "{schema_name}"."{table_name}"'
                    )
                    values = cur.fetchall()
                    values = set(
                        [(table_name, column_name, value[0]) for value in values]
                    )
                    data_tuples.extend(values)

        result = {key: data_tuples}
        results.append(result)

    if rerun:
        client = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL"),
            api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        )

        index_name = index_name.lower()
        client.indices.create(index=index_name, ignore=400)

        for result in results:
            for db_id, data_tuples in result.items():

                def generate_docs():
                    for table_name, column_name, value in data_tuples:
                        yield {
                            "_index": index_name,
                            "_source": {
                                "db_id": db_id,
                                "table": table_name,
                                "column": column_name,
                                "value": value,
                            },
                        }

                helpers.bulk(client, generate_docs())

    return results
