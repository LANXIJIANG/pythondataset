# 02clone.py
import os
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
REPO_LIST_FILE = "python_repos.txt"  # File containing clone URLs from 01Search.py
CLONE_DIR = Path("F:/PythonDataset/repos")  # Directory to clone repositories into
MAX_WORKERS = 4  # Number of parallel cloning threads
RETRY_ATTEMPTS = 3  # Number of retries for failed clones
RETRY_DELAY = 5  # Delay between retries in seconds
LOG_FILE = "F:/PythonDataset/clone.log"  # Log file for cloning process

# Ensure clone directory exists
CLONE_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def read_repo_urls(filename):
    """Read repository clone URLs from the file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(urls)} repository URLs from {filename}")
        return urls
    except FileNotFoundError:
        logger.error(f"Repository list file {filename} not found!")
        raise
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise


def get_repo_name_from_url(url):
    """Extract repository name from clone URL."""
    # Example URL: https://github.com/user/repo.git
    repo_name = url.rstrip(".git").split("/")[-1]
    return repo_name


def clone_repository(url, clone_dir):
    """Clone a single repository with retry mechanism."""
    repo_name = get_repo_name_from_url(url)
    repo_path = clone_dir / repo_name
    if repo_path.exists():
        logger.info(f"Repository {repo_name} already exists at {repo_path}, skipping...")
        return repo_name, True

    for attempt in range(RETRY_ATTEMPTS):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(repo_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Successfully cloned {repo_name} to {repo_path}")
            return repo_name, True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed to clone {repo_name}: {e.stderr}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed to clone {repo_name}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    logger.error(f"Failed to clone {repo_name} after {RETRY_ATTEMPTS} attempts")
    return repo_name, False


def clone_repositories(urls, clone_dir):
    """Clone all repositories in parallel with progress tracking."""
    total_repos = len(urls)
    successful_clones = 0
    failed_clones = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create futures for cloning each repository
        futures = {executor.submit(clone_repository, url, clone_dir): url for url in urls}

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=total_repos, desc="Cloning repositories"):
            repo_name, success = future.result()
            if success:
                successful_clones += 1
            else:
                failed_clones.append(repo_name)

    logger.info(f"Cloning completed: {successful_clones}/{total_repos} repositories cloned successfully")
    if failed_clones:
        logger.warning(f"Failed to clone {len(failed_clones)} repositories: {', '.join(failed_clones)}")
    return successful_clones, failed_clones


def main():
    """Main function to orchestrate the cloning process."""
    logger.info("Starting repository cloning process...")

    # Read repository URLs
    repo_urls = read_repo_urls(REPO_LIST_FILE)

    # Clone repositories
    successful, failed = clone_repositories(repo_urls, CLONE_DIR)

    # Summary
    logger.info(f"Cloning process finished. Summary:")
    logger.info(f"Total repositories: {len(repo_urls)}")
    logger.info(f"Successfully cloned: {successful}")
    logger.info(f"Failed to clone: {len(failed)}")
    if failed:
        logger.info(f"Failed repositories: {', '.join(failed)}")


if __name__ == "__main__":
    main()