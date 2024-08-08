import jinja2
from expand_langchain.utils.registry import chain_registry
from expand_langchain.utils.sampling import sampling_chain
from langchain_core.runnables import RunnableLambda


@chain_registry(name="apply_template")
def apply_template_chain(
    key: str,
    template_path: str,
    **kwargs,
):
    async def _func(data, config={}):
        template = open(template_path).read()
        template = jinja2.Template(template)
        result = template.render(data)

        return {key: result}

    chain = RunnableLambda(_func)

    result = sampling_chain(chain, 1, **kwargs)
    result.name = key

    return result
