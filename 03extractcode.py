# 03extractcode.py
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re

# Configuration
REPO_DIR = Path("F:/PythonDataset/repos")  # Directory containing cloned repositories
EXTRACT_DIR = Path("F:/PythonDataset/extracted_python_code")  # Directory to store extracted Python files
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB size limit for files
MAX_WORKERS = 4  # Number of parallel extraction threads
LOG_FILE = "F:/PythonDataset/extract.log"  # Log file for extraction process
MIN_CODE_LINES = 5  # Minimum lines of non-comment code to consider a file non-trivial

# Ensure extraction directory exists
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more visibility
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    """Compute SHA256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_trivial_file(file_path):
    """Determine if a file is trivial (empty, only comments, or minimal content)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Remove comments and whitespace
        code_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        return len(code_lines) < MIN_CODE_LINES
    except Exception as e:
        logger.warning(f"Failed to read {file_path} to check if trivial: {e}")
        return True  # Treat unreadable files as trivial to skip them

def extract_python_files(repo_path, extract_dir, global_hashes):
    """Extract Python files from a single repository."""
    repo_name = repo_path.name
    extracted_files = 0
    skipped_files = 0
    skipped_size = 0
    skipped_trivial_init = 0
    skipped_trivial_other = 0
    duplicate_files = 0
    repo_hashes = set()  # Track file hashes within this repository

    # Create a subdirectory for the repository in the extraction directory
    repo_extract_dir = extract_dir / repo_name
    repo_extract_dir.mkdir(exist_ok=True)

    # Walk through the repository directory
    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if not file_name.endswith(".py"):
                continue  # Skip non-Python files

            file_path = Path(root) / file_name

            try:
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > MAX_FILE_SIZE:
                    logger.warning(f"Skipping {file_path}: File size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE} bytes)")
                    skipped_files += 1
                    skipped_size += 1
                    continue

                # Skip empty or trivial __init__.py files
                if file_name == "__init__.py":
                    if is_trivial_file(file_path):
                        logger.debug(f"Skipping {file_path}: Empty or trivial __init__.py")
                        skipped_files += 1
                        skipped_trivial_init += 1
                        continue

                # Compute file hash to detect duplicates
                file_hash = get_file_hash(file_path)

                # Check for duplicates within the repository
                if file_hash in repo_hashes:
                    # If the file is trivial, skip it; otherwise, keep it and log a warning
                    if is_trivial_file(file_path):
                        logger.debug(f"Skipping {file_path}: Duplicate and trivial file within {repo_name}")
                        duplicate_files += 1
                        skipped_trivial_other += 1
                        continue
                    else:
                        logger.warning(f"Keeping {file_path}: Duplicate content within {repo_name} but not trivial")
                else:
                    repo_hashes.add(file_hash)

                # Check for duplicates across repositories
                if file_hash in global_hashes:
                    if is_trivial_file(file_path):
                        logger.debug(f"Skipping {file_path}: Duplicate and trivial file across repositories")
                        duplicate_files += 1
                        skipped_trivial_other += 1
                        continue
                    else:
                        logger.warning(f"Keeping {file_path}: Duplicate content across repositories but not trivial")
                else:
                    global_hashes.add(file_hash)

                # Construct destination path
                relative_path = file_path.relative_to(repo_path)
                dest_path = repo_extract_dir / relative_path

                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copy2(file_path, dest_path)
                extracted_files += 1
                logger.debug(f"Extracted {file_path} to {dest_path}")

            except (IOError, OSError) as e:
                logger.warning(f"Failed to extract {file_path}: {e}")
                skipped_files += 1
                continue

    logger.info(f"Processed {repo_name}: Extracted {extracted_files} files, Skipped {skipped_files} files (Size: {skipped_size}, Trivial __init__.py: {skipped_trivial_init}, Other Trivial: {skipped_trivial_other}), Duplicates {duplicate_files}")
    return repo_name, extracted_files, skipped_files, duplicate_files, skipped_size, skipped_trivial_init, skipped_trivial_other

def extract_all_repositories(repo_dir, extract_dir):
    """Extract Python files from all repositories in parallel."""
    # Get list of repositories (directories in repo_dir)
    repos = [repo_dir / repo_name for repo_name in os.listdir(repo_dir) if (repo_dir / repo_name).is_dir()]
    total_repos = len(repos)
    total_extracted = 0
    total_skipped = 0
    total_skipped_size = 0
    total_skipped_trivial_init = 0
    total_skipped_trivial_other = 0
    total_duplicates = 0
    failed_repos = []
    global_hashes = set()  # Track file hashes across all repositories

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create futures for extracting files from each repository
        futures = {executor.submit(extract_python_files, repo, extract_dir, global_hashes): repo for repo in repos}

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=total_repos, desc="Extracting Python files"):
            try:
                repo_name, extracted, skipped, duplicates, skipped_size, skipped_trivial_init, skipped_trivial_other = future.result()
                total_extracted += extracted
                total_skipped += skipped
                total_skipped_size += skipped_size
                total_skipped_trivial_init += skipped_trivial_init
                total_skipped_trivial_other += skipped_trivial_other
                total_duplicates += duplicates
            except Exception as e:
                repo_name = futures[future].name
                logger.error(f"Failed to process {repo_name}: {e}")
                failed_repos.append(repo_name)

    logger.info(f"Extraction completed: Processed {total_repos} repositories")
    logger.info(f"Total files extracted: {total_extracted}")
    logger.info(f"Total files skipped: {total_skipped} (Size: {total_skipped_size}, Trivial __init__.py: {total_skipped_trivial_init}, Other Trivial: {total_skipped_trivial_other})")
    logger.info(f"Total duplicates skipped: {total_duplicates}")
    if failed_repos:
        logger.warning(f"Failed to process {len(failed_repos)} repositories: {', '.join(failed_repos)}")
    return total_extracted, total_skipped, total_duplicates, failed_repos

def main():
    """Main function to orchestrate the extraction process."""
    logger.info("Starting Python code extraction process...")

    # Extract Python files from all repositories
    total_extracted, total_skipped, total_duplicates, failed_repos = extract_all_repositories(REPO_DIR, EXTRACT_DIR)

    # Summary
    logger.info(f"Extraction process finished. Summary:")
    logger.info(f"Total repositories processed: {len(os.listdir(REPO_DIR))}")
    logger.info(f"Total Python files extracted: {total_extracted}")
    logger.info(f"Total files skipped (due to size or errors): {total_skipped}")
    logger.info(f"Total duplicates skipped: {total_duplicates}")
    if failed_repos:
        logger.info(f"Failed repositories: {', '.join(failed_repos)}")

if __name__ == "__main__":
    main()