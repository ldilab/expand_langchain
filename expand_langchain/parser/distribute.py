import re
from typing import Dict, List, Tuple, Union

from expand_langchain.utils.registry import parser_registry
from langchain_core.runnables import RunnableLambda


@parser_registry(name="distribute")
def distribute_runner(landmarks: List[Tuple[str, str]]):
    def func(text: str) -> Dict[str, str]:
        _landmarks = {key: landmark for key, landmark in landmarks}

        # find landmarks
        cursors = {}
        for key, landmark in _landmarks.items():
            match = re.search(landmark, text)
            start = match.start() if match else len(text)
            end = match.end() if match else len(text)
            cursors[key] = (start, end)

        # extract strings
        result = {}
        for key, (start, end) in cursors.items():
            # find nearest other start
            next_starts = [v[0] for k, v in cursors.items() if k != key]
            next_starts.append(len(text))
            next_starts = [v for v in next_starts if v > start]
            _end = min(next_starts, default=len(text))

            # extract string
            result[key] = text[end:_end].rstrip()

        if result == {}:
            result = {"default": text}

        return result

    return RunnableLambda(func, "distribute")


@parser_registry(name="find")
def find_runner(patterns: List[str]):
    def func_str(text: str) -> List[str]:
        """
        find all occurrences of patterns
        the order of result is the order of occurrence in the original text
        """
        result = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                cursor, string = match.start(), match.group()
                result.append((cursor, string))

        result = [string for cursor, string in sorted(result)]

        return result

    def func(
        input: Union[str, Dict[str, str]]
    ) -> Union[List[str], Dict[str, List[str]]]:
        if isinstance(input, str):
            return func_str(input)
        else:
            return {key: func_str(value) for key, value in input.items()}

    return RunnableLambda(func, "find")


@parser_registry(name="join")
def join_runner(landmarks: List[Tuple[str, str]]):
    def func(input: Dict[str, Union[str, List[str]]]) -> str:
        """
        join strings with landmarks
        """
        result = []
        ripple = ""
        for key, landmark in landmarks:
            ripple += landmark
            _result = []
            if key in input and input[key].strip() != "":
                if isinstance(input[key], str):
                    _result.append(input[key])
                else:
                    _result.extend(input[key])
                result.append(ripple + "\n".join(_result))
                ripple = ""

        if ripple:
            result.append(ripple)

        return "\n\n".join(result)

    return RunnableLambda(func, "join")
