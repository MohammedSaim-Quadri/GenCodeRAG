"""
Multi-language function extractor for GitHub scraped code.
Supports Python (AST), and basic heuristic extraction for other languages.
"""
import ast, os, json, re
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/github_raw")
OUTPUT_FILE = Path("data/chunks/github_code_chunks.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# === Extension to Language Map ===
extension_map = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
    '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
    '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
    '.sql': 'sql', '.html': 'html', '.css': 'css', '.sh': 'bash',
    '.m': 'objective-c', '.jl': 'julia', '.lua': 'lua',
    '.dart': 'dart', '.r': 'r', '.pl': 'perl', '.json': 'json'
}

# === Heuristic extractors ===

def extract_python_functions(code):
    """Extract Python functions using AST"""
    try:
        tree = ast.parse(code)
        return [
            {"function_name": node.name, "code": ast.get_source_segment(code, node)}
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
    except Exception:
        return []

def extract_generic_functions(code, lang):
    """
    Heuristic extraction using regex for non-Python languages.
    Not perfect, but filters reusable blocks.
    """
    patterns = {
        'javascript': r"(function\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'typescript': r"(function\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'java': r"(public\s+.*?\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'cpp': r"(\w+\s+\w+\s*\(.*?\)\s*{[\s\S]*?})",
        'c': r"(\w+\s+\w+\s*\(.*?\)\s*{[\s\S]*?})",
        'go': r"(func\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'php': r"(function\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'ruby': r"(def\s+\w+[\s\S]*?end)",
        'swift': r"(func\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'rust': r"(fn\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'csharp': r"(public\s+.*?\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'kotlin': r"(fun\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'scala': r"(def\s+\w+\(.*?\)\s*{[\s\S]*?})",
        'bash': r"(function\s+\w+\s*\(\)\s*{[\s\S]*?})",
    }

    pattern = patterns.get(lang)
    if not pattern:
        return [{"function_name": "default", "code": code}]  # fallback = whole file

    matches = re.findall(pattern, code)
    return [{"function_name": f"func_{i}", "code": match} for i, match in enumerate(matches)]


def build_chunk_id(language, path, func_name):
    """
    Builds a unique chunk ID string across languages.
    E.g., python__public-apis__scripts__validate__format__check_auth
    """
    path_clean = str(path).replace("/", "__").replace("\\", "__").replace(".", "_")
    return f"{language}__{path_clean}__{func_name}"


def process_all():
    # === Main Processing ===

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for lang_dir in RAW_DIR.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name.lower()
            print(f"🔍 Processing language: {lang}")
            for file in tqdm(list(lang_dir.glob("*"))):
                ext = file.suffix
                if ext not in extension_map:
                    continue
                true_lang = extension_map[ext]
                code = file.read_text(encoding="utf-8", errors="ignore")
                functions = []

                if true_lang == "python":
                    functions = extract_python_functions(code)
                else:
                    functions = extract_generic_functions(code, true_lang)

                for func in functions:
                    chunk_id = build_chunk_id(true_lang, file.name, func["function_name"])
                    entry = {
                        "chunk_id":chunk_id,
                        "language": true_lang,
                        "repo": file.name.split("__")[0],
                        "path": file.name,
                        "function_name": func["function_name"],
                        "code": func["code"]
                    }
                    json.dump(entry, out_file, ensure_ascii=False)
                    out_file.write("\n")

if __name__ == "__main__":
    process_all()