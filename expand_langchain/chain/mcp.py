from typing import List

from expand_langchain.utils.registry import chain_registry
from langchain_core.runnables import RunnableLambda
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@chain_registry(name="mcp")
def mcp_chain(
    key: str,
    tool_args_key: str,
    tool_name: str = None,
    tool_name_key: str = None,
    server_kwargs: dict = {},
    **kwargs,
):
    server_params = StdioServerParameters(**server_kwargs)

    async def _func(data, config={}):
        result = {}
        tool_args = data.get(tool_args_key, [{}])[0]
        if tool_name is None:
            _tool_name = data.get(tool_name_key, [""])[0]
        else:
            _tool_name = tool_name

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.call_tool(
                        name=_tool_name, arguments=tool_args
                    )
                    response = response.content[0].text
        except Exception as e:
            response = str(e)

        result = {key: response}

        return result

    return RunnableLambda(_func, name=key)
