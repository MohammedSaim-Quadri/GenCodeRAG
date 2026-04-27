"""
Multi-language GitHub scraper.
Downloads top-starred files by extension and saves them grouped by language.
"""
import os, requests, time
from dotenv import load_dotenv
from pathlib import Path
from config import EXTENSION_MAP
from logger import setup_logger
from settings import settings
logger = setup_logger(__name__)

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise EnvironmentError("GITHUB_TOKEN environment variable is not set")

HEADERS = {
    "Authorization": f"token {TOKEN}"
}
SAVE_DIR = Path("data/github_raw/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def search_repos(
    language,
    license='mit',
    stars=settings.STARS,
    per_page=settings.PER_PAGE,
    page=1
):
    url = "https://api.github.com/search/repositories"

    params = {
        "q": f"language:{language} license:{license} stars:>={stars}",
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page
    }

    max_retries = 3

    for attempt in range(max_retries):
        try:
            res = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30
            )

            if res.status_code == 200:
                return res.json().get("items", [])

            elif res.status_code == 403:
                wait_time = 60 * (attempt + 1)
                print(f"🚫 Rate limit hit. Sleeping for {wait_time} seconds...")
                logger.warning(f"GitHub API rate limit hit. Attempt {attempt + 1} of {max_retries}. Sleeping for {wait_time} seconds.")
                time.sleep(wait_time)

            else:
                print(f"❌ GitHub API error {res.status_code}: {res.text}")
                logger.error(f"GitHub API error {res.status_code}: {res.text}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            logger.error(f"Request failed: {e}")
            return []

    print("❌ Max retries exceeded while searching repositories.")
    logger.error("Max retries exceeded while searching repositories.")
    return []


def get_repo_files(owner, repo):
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"

    max_retries = 3

    for attempt in range(max_retries):
        try:
            res = requests.get(
                tree_url,
                headers=HEADERS,
                timeout=30
            )

            if res.status_code == 200:
                return res.json().get("tree", [])

            elif res.status_code == 403:
                wait_time = 60 * (attempt + 1)
                print(f"🚫 Rate limit hit during file fetch. Sleeping for {wait_time} seconds...")
                logger.warning(f"GitHub API rate limit hit during file fetch. Attempt {attempt + 1} of {max_retries}. Sleeping for {wait_time} seconds.")
                time.sleep(wait_time)

            else:
                print(f"❌ Failed to fetch repo files: {res.status_code}")
                logger.error(f"Failed to fetch repo files: {res.status_code} - {res.text}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ Repo file fetch failed: {e}")
            logger.error(f"Repo file fetch failed: {e}")
            return []

    print(f"❌ Max retries exceeded while fetching files for {owner}/{repo}")
    logger.error(f"Max retries exceeded while fetching files for {owner}/{repo}")
    return []

def download_file(owner, repo, path, lang_folder):
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    res = requests.get(raw_url, timeout=30)
    if res.status_code == 200 and len(res.content) < 20000:
        save_path = SAVE_DIR / lang_folder
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / f"{owner}__{repo}__{path.replace('/', '__')}"
        out_file.write_text(res.text, encoding="utf-8")
        logger.info(f"Downloaded file: {owner}/{repo}/{path}")
    else:
        print(f"⚠️ Failed to download {owner}/{repo}/{path} (status: {res.status_code}, size: {len(res.content)})")
        logger.warning(f"Failed to download {owner}/{repo}/{path} (status: {res.status_code}, size: {len(res.content)})")

def is_language_done(lang_folder, required_repos=settings.MAX_REPOS):
    lang_path = SAVE_DIR / lang_folder
    if not lang_path.exists():
        return False
    files = list(lang_path.glob("*"))
    # Assume at least 1 file per repo, deduplicate by repo
    repo_ids = set(f.name.split("__")[1] for f in files if "__" in f.name)
    return len(repo_ids) >= required_repos

def run():
    for ext, lang in EXTENSION_MAP.items():
        lang_folder = lang.lower()

        if is_language_done(lang_folder):
            print(f"⏩ Skipping {lang} (already has ≥{settings.MAX_REPOS} repos scraped)")
            logger.info(f"Skipping {lang} - already has ≥{settings.MAX_REPOS} repos scraped.")
            continue

        print(f"\nScraping {settings.MAX_REPOS} repos for: {lang}")
        logger.info(f"Starting scrape for {lang} - target {settings.MAX_REPOS} repos.")
        collected = []
        seen_repos = set()
        page = 1

        empty_retry_count = 0
        max_empty_retries = 5

        while len(collected) < settings.MAX_REPOS:
            repos = search_repos(
                    language=lang,
                    stars=settings.STARS,
                    per_page=settings.PER_PAGE,
                    page=page
                )
            
            if not repos:
                empty_retry_count += 1

                if empty_retry_count >= max_empty_retries:
                    print(f"Stopping {lang}: no repos found after multiple attempts.")
                    logger.warning(f"Stopping {lang} - no repos found after {max_empty_retries} attempts.")
                    break

                continue

            for repo in repos:
                full_name = repo["full_name"]
                if full_name not in seen_repos:
                    seen_repos.add(full_name)
                    collected.append(repo)
                    if len(collected) >= settings.MAX_REPOS:
                        break
            page += 1
            time.sleep(1)

        print(f"Collected {len(collected)} repos for {lang}")
        logger.info(f"Collected {len(collected)} repos for {lang} after searching {page - 1} pages.")

        for repo in collected:
            owner, name = repo["owner"]["login"], repo["name"]
            print(f"Processing: {owner}/{name}")
            logger.info(f"Processing repository: {owner}/{name}")
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
                            if count >= settings.MAX_FILES_PER_REPO:
                                break
                        except Exception as e:
                            print("Download failed:", fpath, e)
                            logger.warning(f"Download failed for {owner}/{name}/{fpath}: {e}")
            except Exception as e:
                print("Failed to fetch repo files:", name, e)
                logger.warning(f"Failed to fetch files for {owner}/{name}: {e}")
        
    # === Print Summary ===
    print("\n📊 Scraped Language Summary:")
    logger.info("Scraping completed. Summary of scraped languages:")
    for ext, lang in EXTENSION_MAP.items():
        lang_path = SAVE_DIR / lang.lower()
        if lang_path.exists():
            files = list(lang_path.glob("*"))
            repos = set(f.name.split("__")[1] for f in files if "__" in f.name)
            print(f"- {lang}: {len(repos)} repos, {len(files)} files")

if __name__ == "__main__":
    run()