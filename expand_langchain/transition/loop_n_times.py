from expand_langchain.utils.registry import transition_registry
from langgraph.graph import END


@transition_registry(name="loop_n_times")
def loop_n_times(
    dest: str,
    dest2: str = END,
    n: int = 1,
):
    def func(state):
        if len(state) < n + 1:
            return dest
        else:
            return dest2

    return func
