"""
Multi-language function extractor for GitHub scraped code.
Supports Python (AST), and basic heuristic extraction for other languages.
"""
import ast, os, json, re
from pathlib import Path
from tqdm import tqdm
import hashlib
from config import EXTENSION_MAP
from pipeline.parser_utils import extract_javascript_functions
from logger import setup_logger
logger = setup_logger(__name__)

RAW_DIR = Path("data/github_raw")
OUTPUT_FILE = Path("data/chunks/github_code_chunks.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# === Heuristic extractors ===

def extract_python_functions(code):
    """Extract Python functions using AST"""
    try:
        tree = ast.parse(code)
        return [
            {"function_name": node.name, "code": ast.get_source_segment(code, node)}
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except Exception:
        return []

def extract_generic_functions(code, lang):
    """
    Heuristic extraction using regex for non-Python languages.
    Not perfect, but filters reusable blocks.
    """
    if lang == "javascript":
        return extract_javascript_functions(code)
    
    patterns = {
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
        if len(code) < 5000:
            return [{
                "function_name": "default",
                "code": code
            }]

        return []

    matches = re.findall(pattern, code)
    return [{"function_name": f"func_{i}", "code": match} for i, match in enumerate(matches)]


def build_chunk_id(language, repo, path, func_name, code):
    raw_string = f"{language}_{repo}_{path}_{func_name}_{code[:200]}"
    short_hash = hashlib.md5(raw_string.encode()).hexdigest()[:10]

    return f"{language}__{repo}__{func_name}__{short_hash}"


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
                if ext not in EXTENSION_MAP:
                    continue
                true_lang = EXTENSION_MAP[ext]
                try:
                    code = file.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeDecodeError) as e:
                    print(f"⚠️ Skipping file due to read error: {file} ({e})")
                    continue
                functions = []

                if true_lang == "python":
                    functions = extract_python_functions(code)
                    if not functions:
                        print(f"⚠️ No functions extracted from: {file.name}")
                        continue
                else:
                    functions = extract_generic_functions(code, true_lang)
                    if not functions:
                        print(f"⚠️ No functions extracted from: {file.name}")
                        continue

                for func in functions:
                    repo_name = file.name.split("__")[0]

                    chunk_id = build_chunk_id(
                        true_lang,
                        repo_name,
                        file.name,
                        func["function_name"],
                        func["code"]
                    )
                    entry = {
                        "chunk_id": chunk_id,
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