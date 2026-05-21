from typing import Any

LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "python": ["flask", "django", "pandas", "numpy", "python", "py"],
    "java": ["spring", "jdk", "java", "jvm"],
    "javascript": ["node", "express", "js", "javascript", "react"],
    "typescript": ["ts", "typescript", "nestjs"],
    "cpp": ["c++", "cpp", "stl"],
    "c": ["c ", "c-language", "c code"],
    "csharp": ["c#", "dotnet", "csharp"],
    "go": ["golang", "go "],
    "rust": ["rust"],
    "php": ["php", "laravel"],
    "ruby": ["ruby", "rails"],
    "bash": ["shell", "bash", "sh"],
    "kotlin": ["kotlin", "android"],
    "swift": ["swift", "ios"],
    "scala": ["scala"],
    "sql": ["sql", "database", "postgres", "mysql"],
}


def infer_language_from_prompt(prompt: str) -> str | None:
    prompt_lower = prompt.lower()
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for word in keywords:
            if word in prompt_lower:
                return lang
    return None


def format_code_snippet(code: str, max_lines: int = 50) -> str:
    lines = code.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["...", "# Code truncated for brevity"]
    return "\n".join(lines)


def create_enriched_prompt(query: str, chunks: list[Any]) -> str:
    prompt_context = ""
    total_chars = 0
    max_chars = 12_000

    for point in chunks:
        payload = point.payload
        lang = payload.get("language", "text")
        repo = payload.get("repo", "unknown")
        path = payload.get("path", "")
        func = payload.get("function_name", "")
        code = format_code_snippet(payload.get("code", ""))

        snippet = f"### [{lang.upper()}] {repo}/{path}\nFunction: {func}\n```{lang}\n{code}\n```"

        if total_chars + len(snippet) > max_chars:
            break

        prompt_context += snippet
        total_chars += len(snippet)

    safe_query = query.replace("{", "").replace("}", "")

    return f"""You are an expert developer.

Use the following code snippets to generate a solution for the task.

# CODE CONTEXT:
{prompt_context}

# TASK:
{safe_query}

# INSTRUCTIONS:
- Analyze the provided code examples for patterns and best practices
- Generate clean, well-documented, and efficient code
- Include appropriate error handling and edge cases
- Add helpful comments explaining the logic
- Generate high-quality, production-level code

# RESPONSE:
"""