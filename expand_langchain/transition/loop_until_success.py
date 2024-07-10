from langgraph.graph import END

from expand_langchain.utils.registry import transition_registry


@transition_registry(name="loop_until_success")
def loop_until_success(
    dest: str,
    max_iterations: int,
):
    def func(state):
        if len(state[-1]["exec_result_failed"]) == 0:
            return END
        elif len(state) < max_iterations * 2 + 2:
            return dest
        else:
            return END

    return func
