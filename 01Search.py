# 01Search.py
import requests
import time
import os
from itertools import cycle
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load GitHub tokens from .env
load_dotenv("load_env.env")
tokens = os.getenv("GITHUB_TOKENS", "").split(",")
if not tokens or tokens == [""]:
    raise ValueError("No GitHub tokens found! Add them to .env file as GITHUB_TOKENS=token1,token2,...")
print("Tokens loaded:", [token[:10] + "..." for token in tokens])  # Mask tokens for safety

# API endpoints
SEARCH_URL = "https://api.github.com/search/repositories"
CODE_SEARCH_URL = "https://api.github.com/search/code"
headers = {'Authorization': f'Bearer {tokens[0]}', 'Accept': 'application/vnd.github.v3+json'}
token_cycle = cycle(tokens)

# Configuration
MIN_PYTHON_FILES = 5      # Increased for more substantial repos
MAX_PAGES = 10           # Per query
MAX_REPOS = 500          # Target dataset size
MAX_PER_QUERY = 150      # Reduced to balance across queries
REQUESTS_PER_MINUTE = 30 # GitHub code search limit
SECONDS_PER_REQUEST = 60 / REQUESTS_PER_MINUTE

all_repos = {}
processed_repos = 0
request_timestamps = {token: [] for token in tokens}
rate_limits = {token: {'remaining': 30, 'reset': time.time() + 60} for token in tokens}

# Refined search queries for quality
SEARCH_QUERIES = [
    "language:Python stars:10..99 size:>100",        # Small but non-trivial projects
    "language:Python stars:100..2000 size:>500",     # Medium-sized libraries
    "language:Python stars:>2000 size:>1000",        # Large frameworks
]

def update_headers():
    global headers
    next_token = next(token_cycle)
    headers = {'Authorization': f'Bearer {next_token}', 'Accept': 'application/vnd.github.v3+json'}
    print(f"Switched to token: {next_token[:10]}...")

def check_rate_limit(response, token):
    remaining = int(response.headers.get('X-RateLimit-Remaining', 30))
    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
    rate_limits[token] = {'remaining': remaining, 'reset': reset_time}
    print(f"Rate limit for {token[:10]}...: {remaining} requests remaining, resets at {datetime.fromtimestamp(reset_time)}")
    return remaining > 5

def wait_for_rate_limit_reset():
    now = time.time()
    earliest_reset = min(rate_limits[token]['reset'] for token in tokens)
    wait_time = max(earliest_reset - now, 1)
    if wait_time > 0:
        print(f"All tokens rate-limited. Sleeping for {wait_time:.1f} seconds...")
        time.sleep(wait_time)
    return next(token_cycle)

def enforce_rate_limit(token):
    now = datetime.now()
    minute_ago = now - timedelta(minutes=1)
    request_timestamps[token] = [t for t in request_timestamps[token] if t > minute_ago]
    if len(request_timestamps[token]) >= REQUESTS_PER_MINUTE - 5 or rate_limits[token]['remaining'] <= 5:
        next_token = wait_for_rate_limit_reset()
        global headers
        headers = {'Authorization': f'Bearer {next_token}', 'Accept': 'application/vnd.github.v3+json'}
        print(f"Switched to token: {next_token[:10]}...")
    time.sleep(SECONDS_PER_REQUEST)  # Pace requests

def has_enough_python_files(repo_full_name):
    current_token = headers['Authorization'].split()[1]
    python_count_query = f"extension:py repo:{repo_full_name}"
    for attempt in range(3):
        try:
            enforce_rate_limit(current_token)
            response = requests.get(CODE_SEARCH_URL, params={'q': python_count_query, 'per_page': 1}, headers=headers)
            current_token = headers['Authorization'].split()[1]
            request_timestamps[current_token].append(datetime.now())
            if not check_rate_limit(response, current_token):
                continue
            if response.status_code != 200:
                print(f"Code search failed for {repo_full_name}: {response.status_code} - {response.text[:100]}")
                time.sleep(2)
                continue
            data = response.json()
            python_count = data.get("total_count", 0)
            print(f"Python files in {repo_full_name}: {python_count}")
            return python_count >= MIN_PYTHON_FILES
        except requests.RequestException as e:
            print(f"Error counting Python files for {repo_full_name} (attempt {attempt + 1}/3): {e}")
            time.sleep(2)
    print(f"Failed to count Python files for {repo_full_name} after 3 attempts")
    return False

def fetch_repositories():
    global processed_repos
    repos_per_query = {query: 0 for query in SEARCH_QUERIES}
    for query in SEARCH_QUERIES:
        if len(all_repos) >= MAX_REPOS:
            break
        params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': 100, 'page': 1}
        while params["page"] <= MAX_PAGES and len(all_repos) < MAX_REPOS and repos_per_query[query] < MAX_PER_QUERY:
            try:
                print(f"Fetching page {params['page']} for query: {query}")
                current_token = headers['Authorization'].split()[1]
                enforce_rate_limit(current_token)
                response = requests.get(SEARCH_URL, params=params, headers=headers)
                request_timestamps[current_token].append(datetime.now())
                if not check_rate_limit(response, current_token):
                    continue
                if response.status_code != 200:
                    print(f"Repo search failed: {response.status_code} - {response.text[:100]}")
                    break
                data = response.json()
                total_repos = data.get("total_count", 0)
                print(f"Total items found for {query}: {total_repos}")
                repos = data.get("items", [])
                if not repos:
                    print(f"No more repos for {query}")
                    break
                for repo in repos:
                    if len(all_repos) >= MAX_REPOS or repos_per_query[query] >= MAX_PER_QUERY:
                        print(f"Limit reached: {len(all_repos)} total repos, {repos_per_query[query]} for {query}")
                        break
                    processed_repos += 1
                    full_name = repo["full_name"]
                    if repo.get("language") != "Python":
                        print(f"[{processed_repos}/{total_repos}] Skipping {full_name} - not primarily Python")
                        continue
                    print(f"[{processed_repos}/{total_repos}] Checking {full_name}...")
                    if has_enough_python_files(full_name) and full_name not in all_repos:
                        all_repos[full_name] = {"url": repo["html_url"], "clone_url": repo["clone_url"]}
                        repos_per_query[query] += 1
                        print(f"[{processed_repos}/{total_repos}] ✅ Added {full_name} (Query: {query})")
                    else:
                        print(f"[{processed_repos}/{total_repos}] 🔄 Skipped {full_name} (< {MIN_PYTHON_FILES} Python files)")
            except requests.RequestException as e:
                print(f"Error fetching repos for {query}: {e}")
                time.sleep(5)
            params["page"] += 1
            save_to_file()

def save_to_file(filename="python_repos.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for repo in all_repos.values():
                f.write(f"{repo['clone_url']}\n")  # Save clone_url for later cloning
        print(f"✅ Saved {len(all_repos)} repos to {filename}")
    except IOError as e:
        print(f"❌ File error: {e}")

if __name__ == "__main__":
    print("🚀 Fetching Python repositories...")
    fetch_repositories()
    save_to_file()