from model.final_rag_system import infer_language_from_prompt


def test_detect_python():
    query = "Create a Python Flask API for login"

    result = infer_language_from_prompt(query)

    assert result == "python"


def test_detect_java():
    query = "Build a Java Spring Boot application"

    result = infer_language_from_prompt(query)

    assert result == "java"


def test_detect_javascript():
    query = "Create a Node.js Express server"

    result = infer_language_from_prompt(query)

    assert result == "javascript"


def test_detect_sql():
    query = "Write a SQL query for PostgreSQL joins"

    result = infer_language_from_prompt(query)

    assert result == "sql"


def test_no_language_detected():
    query = "Build a login system"

    result = infer_language_from_prompt(query)

    assert result is None