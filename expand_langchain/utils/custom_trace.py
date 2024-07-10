from langsmith import Client
from langsmith import traceable as _traceable


def traceable(*args, hide=False, **kwargs):
    if hide:
        client = Client(hide_inputs=lambda x: {}, hide_outputs=lambda x: {})
    else:
        client = Client()

    def wrapper(func):
        return _traceable(*args, client=client, **kwargs)(func)

    return wrapper
