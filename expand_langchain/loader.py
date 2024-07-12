import json
import logging
import os
from pathlib import Path

import psycopg2
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

            elif dataset.type == "index":
                datasets[name] = _load_index(sources, **dataset.kwargs)

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


def _load_index(sources_dict, mode, sources, path, model, platform):
    if platform == "open_webui":
        embedding_function = OllamaEmbeddings(
            model=model,
            base_url=os.environ["OPEN_WEBUI_BASE_URL"],
            headers={
                "Authorization": f"Bearer {os.environ['OPEN_WEBUI_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
    else:
        raise ValueError(f"Unknown platform: {platform}")

    client = chromadb.PersistentClient(path)

    results = {}
    for key in sources:
        try:
            client.get_collection(key)
        except ValueError:
            collection = client.create_collection(key)
            conn = sources_dict[key]
            values = defaultdict(list)
            if mode == "entity":
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            n.nspname AS schema,
                            c.relname AS table,
                            a.attname AS column,
                            pg_catalog.obj_description(c.oid, 'pg_class') AS table_description,
                            pg_catalog.col_description(a.attrelid, a.attnum) AS column_description
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
                            AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY
                            n.nspname, c.relname, a.attnum;"""
                    )
                    output = cur.fetchall()

                    for (
                        schema,
                        table,
                        column,
                        table_description,
                        column_description,
                    ) in output:
                        cur.execute(
                            f"SELECT DISTINCT {column} FROM {schema}.{table} LIMIT 20;"
                        )
                        rows = cur.fetchall()
                        for row in rows:
                            values[row[0]].append(
                                {
                                    "table": table,
                                    "column": column,
                                    "table_description": table_description,
                                    "column_description": column_description,
                                }
                            )
            elif mode == "context":
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
                            n.nspname, c.relname;"""
                    )
                    output = cur.fetchall()
                    for schema, table, description in output:
                        values[description].append({"table": f"{schema}.{table}"})

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            n.nspname AS schema,
                            c.relname AS table,
                            a.attname AS column,
                            pg_catalog.col_description(a.attrelid, a.attnum) AS description
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
                            AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY
                            n.nspname, c.relname, a.attnum;"""
                    )
                    output = cur.fetchall()
                    for schema, table, column, description in output:
                        values[description].append(
                            {
                                "table": f"{schema}.{table}",
                                "column": column,
                            }
                        )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            collection.add(
                ids=list(map(str, range(len(values)))),
                documents=list(values.keys()),
                metadatas=list(values.values()),
            )

        results[key] = Chroma(
            client=client,
            collection_name=key,
            embedding_function=embedding_function,
        )

    return results


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
