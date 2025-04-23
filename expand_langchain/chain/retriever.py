import os
from typing import List

from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda


@chain_registry(name="retriever")
def retriever_chain(
    key: str,
    method: str,
    query_key: str,
    index_name: str,
    k1: float = 2.0,
    b: float = 0.75,
    **kwargs,
):
    def _func(data, config={}):
        if method == "elasticsearch":
            from elasticsearch import Elasticsearch

            es = Elasticsearch(
                os.environ["ELASTICSEARCH_URL"],
                api_key=os.environ["ELASTICSEARCH_API_KEY"],
            )

            def search(query):
                try:
                    response = es.search(
                        index=index_name,
                        body=query,
                    )
                except Exception as e:
                    return []

                return response["hits"]["hits"]

            retriever = RunnableLambda(search, name="elasticsearch_retriever")

        else:
            raise ValueError(f"Method {method} is not supported")

        result = {}
        result[key] = []
        for input in data[query_key]:
            if isinstance(input, str):
                result = retriever.invoke(input)
                result[key].append(result)
            elif isinstance(input, List):
                results = retriever.batch(input)
                result[key].append(results)
            elif isinstance(input, dict):
                results = retriever.invoke(input)
                result[key].append(results)
            else:
                raise ValueError("Invalid input type")

        return result

    return RunnableLambda(_func, name="retriever")
