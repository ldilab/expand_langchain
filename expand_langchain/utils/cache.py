from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel


class CacheCallbackHandler(BaseModel, BaseCallbackHandler):
    raise_error: bool = True

    current_chain_name: str = None

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        current_chain_name = serialized["name"]

        if self.current_chain_name == "LangGraph":
            raise NotImplementedError
        else:
            pass

    def on_chain_end(
        self,
        outputs,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        if self.current_chain_name == "LangGraph":
            raise NotImplementedError
        else:
            pass
