"""
Multi-language GitHub scraper.
Downloads top-starred files by extension and saves them grouped by language.
"""
import os, requests, time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization":f"token {TOKEN}"}
SAVE_DIR = Path("data/github_raw/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MAX_REPOS = 30       # limit number of repos processed
MAX_FILES_PER_REPO = 10  # limit .py files per repo

# === Supported Extensions ===

extension_map = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
    '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
    '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala','.html': 'html', '.css': 'css', '.sh': 'bash',
    '.m': 'objective-c', '.jl': 'julia', '.lua': 'lua',
    '.dart': 'dart'
}

PER_PAGE = 30
STARS = 400

def search_repos(language, license='mit', stars=STARS, per_page=PER_PAGE):
    url = f"https://api.github.com/search/repositories"
    params = {
        "q": f"language:{language} license:{license} stars:>={stars}",
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": 1
    }
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code == 403:
        print("🚫 Rate limit hit. Sleeping for 60 minutes...")
        time.sleep(3600)
        return search_repos(language, license, stars, per_page)
    if res.status_code == 200:
        return res.json().get("items", [])
    else:
        print(f"❌ GitHub API error {res.status_code}: {res.text}")
        return []

def get_repo_files(owner, repo):
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    res = requests.get(tree_url, headers=HEADERS)
    if res.status_code == 403:
        print("🚫 Rate limit hit during file fetch. Sleeping for 60 minutes...")
        time.sleep(3600)
        return get_repo_files(owner, repo)
    if res.status_code == 200:
        return res.json().get("tree", [])
    return []

def download_file(owner, repo, path, lang_folder):
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    res = requests.get(raw_url)
    if res.status_code == 200 and len(res.text) < 20000:
        save_path = SAVE_DIR / lang_folder
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / f"{owner}__{repo}__{path.replace('/', '__')}"
        out_file.write_text(res.text, encoding="utf-8")

def is_language_done(lang_folder, required_repos=MAX_REPOS):
    lang_path = SAVE_DIR / lang_folder
    if not lang_path.exists():
        return False
    files = list(lang_path.glob("*"))
    # Assume at least 1 file per repo, deduplicate by repo
    repo_ids = set(f.name.split("__")[1] for f in files if "__" in f.name)
    return len(repo_ids) >= required_repos

def run():
    for ext, lang in extension_map.items():
        lang_folder = lang.lower()

        if is_language_done(lang_folder):
            print(f"⏩ Skipping {lang} (already has ≥{MAX_REPOS} repos scraped)")
            continue

        print(f"\nScraping {MAX_REPOS} repos for: {lang}")
        collected = []
        page = 1

        while len(collected) < MAX_REPOS:
            repos = search_repos(language=lang, stars=STARS, per_page=PER_PAGE)
            if not repos:
                continue
            for repo in repos:
                full_name = repo["full_name"]
                if full_name not in [r["full_name"] for r in collected]:
                    collected.append(repo)
                    if len(collected) >= MAX_REPOS:
                        break
            page += 1
            time.sleep(1)

        print(f"Collected {len(collected)} repos for {lang}")

        for repo in collected:
            owner, name = repo["owner"]["login"], repo["name"]
            print(f"Processing: {owner}/{name}")
            try:
                files = get_repo_files(owner, name)
                count = 0
                for file in files:
                    fpath = file.get("path", "")
                    if fpath.endswith(ext):
                        try:
                            download_file(owner, name, fpath, lang)
                            count += 1
                            time.sleep(0.4)
                            if count >= MAX_FILES_PER_REPO:
                                break
                        except Exception as e:
                            print("Download failed:", fpath, e)
            except Exception as e:
                print("Failed to fetch repo files:", name, e)
        
    # === Print Summary ===
    print("\n📊 Scraped Language Summary:")
    for ext, lang in extension_map.items():
        lang_path = SAVE_DIR / lang.lower()
        if lang_path.exists():
            files = list(lang_path.glob("*"))
            repos = set(f.name.split("__")[1] for f in files if "__" in f.name)
            print(f"- {lang}: {len(repos)} repos, {len(files)} files")

if __name__ == "__main__":
    run()