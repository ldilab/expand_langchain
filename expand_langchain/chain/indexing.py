import os
from typing import List

from expand_langchain.utils.registry import chain_registry, model_registry
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_elasticsearch import ElasticsearchStore
from typing import *


@chain_registry(name="indexing")
def elastic_index_chain(
    index_name: str,
    embedding_model: Dict[str, Any] = {},
    page_content_key: str = "text",
    id_key: str = "id",
    **kwargs,
):
    def _func(data, config={}):
        embedding_model_obj = model_registry["embedding"](**embedding_model)
        while True:
            try:
                es = ElasticsearchStore(
                    index_name=index_name,
                    embedding=embedding_model_obj,
                    es_url=os.environ["ELASTICSEARCH_URL"],
                    es_api_key=os.environ["ELASTICSEARCH_API_KEY"],
                )
                break
            except:
                print("Error connecting to Elasticsearch. Retrying...")
                continue

        target_page_contents = data[page_content_key]
        # flatten to 1d list
        while isinstance(target_page_contents, list) and isinstance(target_page_contents[0], list):
            target_page_contents = [item for sublist in target_page_contents for item in sublist]

        data_documents = []
        for idx, page_content in enumerate(target_page_contents):
            data_document = Document(
                page_content=page_content,
                metadata={"source": f"{data[id_key]};{idx}", "id": data[id_key], "idx": idx},
            )
            data_documents.append(data_document)

        while True:
            try:
                result = es.add_documents(
                    documents=data_documents,
                )
                break
            except Exception as e:
                print(f"Error indexing document: {e}")
                continue

        return {"indexing_result": result}

    return RunnableLambda(_func, name="indexing")
