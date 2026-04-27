from pipeline.parser_utils import extract_javascript_functions


def test_extract_javascript_function():
    code = """
function greet() {
    console.log("hello");
}
"""

    result = extract_javascript_functions(code)

    assert len(result) == 1
    assert result[0]["function_name"] == "greet"