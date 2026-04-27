from pipeline.clean_code import (
    extract_python_functions,
    extract_generic_functions,
)


def test_extract_python_function():
    code = """
def hello():
    print("hello")
"""

    result = extract_python_functions(code)

    assert len(result) == 1
    assert result[0]["function_name"] == "hello"


def test_extract_async_python_function():
    code = """
async def fetch_data():
    return "done"
"""

    result = extract_python_functions(code)

    assert len(result) == 1
    assert result[0]["function_name"] == "fetch_data"


def test_extract_javascript_function():
    code = """
function greet() {
    console.log("hello");
}
"""

    result = extract_generic_functions(code, "javascript")

    assert len(result) >= 1


def test_large_unknown_file_returns_empty():
    code = "x" * 6000

    result = extract_generic_functions(code, "unknown")

    assert result == []


def test_small_unknown_file_returns_default():
    code = "print('hello world')"

    result = extract_generic_functions(code, "unknown")

    assert len(result) == 1
    assert result[0]["function_name"] == "default"