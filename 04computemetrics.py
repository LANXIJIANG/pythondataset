import json
import logging
import os
import ast
import time
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import lizard
from tqdm import tqdm
from charset_normalizer import detect

# Configuration
CONFIG_DEFAULTS = {
    "paths": {
        "extract_dir": "F:/PythonDataset/extracted_python_code",
        "output_json": "F:/PythonDataset/04metrics/code_metrics.jsonl",
        "log_file": "F:/PythonDataset/04metrics/analysis.log"
    },
    "limits": {
        "max_workers": 6,
        "max_file_size": 2_000_000,
        "max_lines": 15_000,
        "batch_size": 100,
        "snippet_max_length": 20000,
        "sample_size": 1.0
    },
    "version": "v2.26.4"
}

CONFIG = CONFIG_DEFAULTS
EXTRACT_DIR = Path(CONFIG["paths"]["extract_dir"])
OUTPUT_JSON = Path(CONFIG["paths"]["output_json"])
MAX_WORKERS = CONFIG["limits"]["max_workers"]
VERSION = CONFIG["version"]

os.makedirs(OUTPUT_JSON.parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["paths"]["log_file"], mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_file_content(file_path: Path, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            detection = detect(raw_data)
            encoding = detection['encoding'] if detection['encoding'] else 'latin1'
            confidence = detection['confidence'] if detection['confidence'] else 0.0
            logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence})")
            try:
                return raw_data.decode(encoding, errors='replace')
            except (UnicodeDecodeError, TypeError):
                logger.warning(f"Failed to decode {file_path} with {encoding}, falling back to latin1")
                return raw_data.decode('latin1', errors='replace')
        except (PermissionError, FileNotFoundError, IOError) as e:
            if attempt == retries - 1:
                logger.error(f"Failed to read {file_path} after {retries} attempts: {e}")
            time.sleep(2 ** attempt)
    return None

def preprocess_code(file_path: Path) -> Optional[str]:
    code_content = read_file_content(file_path)
    if not code_content or not code_content.strip():
        logger.warning(f"Skipping {file_path}: empty or invalid content")
        return None
    lines = code_content.splitlines()
    if len(lines) > CONFIG["limits"]["max_lines"]:
        logger.warning(f"Skipping {file_path} due to excessive lines: {len(lines)}")
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            ast.parse(code_content)
        return code_content
    except (SyntaxError, TypeError, ValueError) as e:
        logger.warning(f"Skipping {file_path} due to syntax error: {e}")
        return None

def count_fan_out(node):
    """Count outgoing function calls within a node."""
    fan_out = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            fan_out += 1
    return fan_out

def compute_lizard_metrics(file_path: Path, code_content: str) -> Dict[str, Dict]:
    results = {}
    if code_content is None:
        return results  # Skip if code_content is invalid
    try:
        analysis = lizard.analyze_file(str(file_path))
        tree = ast.parse(code_content)

        method_params = {}
        method_nodes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name
                nop = len(node.args.args)
                nest = max((compute_nesting_depth(child) for child in node.body if
                            isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try))), default=0)
                method_params[method_name] = (nop, nest)
                method_nodes[method_name] = node

        for func in analysis.function_list:
            method_name = func.name.split('.')[-1]
            nop = func.parameter_count or method_params.get(method_name, (0, 0))[0]
            nest = method_params.get(method_name, (0, 0))[1]
            fan_out = count_fan_out(method_nodes.get(method_name, ast.parse("pass")))
            results[method_name] = {
                "NOP_method": nop,
                "LOC_method": func.nloc,
                "CYCLO_method": min(func.cyclomatic_complexity, 25),
                "NEST": nest,
                "TOKEN_COUNT": func.token_count,
                "LENGTH": func.length,
                "FAN_IN": 0,
                "FAN_OUT": fan_out,
                "start_line": func.start_line,
                "end_line": func.end_line,
                "LOC": 0,
                "NOM": 0,
                "ATTR_COUNT": 0,
                "INHERIT_DEPTH": 0
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name in results:
                    continue
                start_line = node.lineno
                end_line = max((n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')), default=start_line)
                attr_count = len(
                    [n for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)])
                inherit_depth = len(node.bases) + max(
                    (compute_inherit_depth(b) for b in node.bases if isinstance(b, ast.Name)), default=0)
                fan_out = count_fan_out(node)
                results[class_name] = {
                    "LOC": end_line - start_line + 1,
                    "NOM": len([n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]),
                    "start_line": start_line,
                    "end_line": end_line,
                    "LOC_method": 0,
                    "NOP_method": 0,
                    "CYCLO_method": 0,
                    "NEST": 0,
                    "TOKEN_COUNT": sum(func.token_count for func in analysis.function_list if
                                       func.start_line >= start_line and func.end_line <= end_line),
                    "LENGTH": end_line - start_line + 1,
                    "FAN_IN": 0,
                    "FAN_OUT": fan_out,
                    "ATTR_COUNT": attr_count,
                    "INHERIT_DEPTH": inherit_depth
                }

        if not results:
            lines = code_content.splitlines()
            results["whole_file"] = {
                "NOP_method": 0,
                "LOC_method": len(lines),
                "CYCLO_method": 1,
                "NEST": 0,
                "TOKEN_COUNT": sum(func.token_count for func in analysis.function_list),
                "LENGTH": len(lines),
                "FAN_IN": 0,
                "FAN_OUT": count_fan_out(tree),
                "start_line": 1,
                "end_line": len(lines),
                "LOC": 0,
                "NOM": 0,
                "ATTR_COUNT": 0,
                "INHERIT_DEPTH": 0
            }
    except Exception as e:
        logger.error(f"Error computing metrics for {file_path}: {e}")
    return results

def compute_nesting_depth(node):
    if not isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
        return 0
    max_depth = 0
    for child in ast.iter_child_nodes(node):
        depth = compute_nesting_depth(child)
        max_depth = max(max_depth, depth)
    return max_depth + 1

def compute_inherit_depth(node):
    if not isinstance(node, ast.Name):
        return 0
    return 1  # Simplified

def extract_smell_code(code_content: str, start_line: int, end_line: int, smell_type: str, metrics: Dict) -> str:
    try:
        lines = code_content.splitlines()
        total_lines = len(lines)
        if not lines or start_line < 1 or end_line > total_lines:
            start_line = max(1, min(start_line, total_lines))
            end_line = min(end_line, total_lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)

        # Default: Extract the segment
        snippet = lines[start_idx:end_idx]

        # Specific handling for each smell type
        if smell_type == "LongParameterList":
            # Extract only the method signature without parsing
            for idx, line in enumerate(snippet):
                if line.strip().startswith("def"):
                    # Find the end of the signature (up to the colon)
                    sig_lines = [line]
                    for j in range(idx + 1, len(snippet)):
                        current_line = snippet[j]
                        sig_lines.append(current_line)
                        if ":" in current_line:
                            break
                    snippet = sig_lines
                    break
            else:
                # If no method definition is found, use a limited portion of the snippet
                snippet = snippet[:5]
        elif smell_type == "LargeClass":
            # Allow more lines for LargeClass to capture structure
            if len(snippet) > 400:
                snippet = snippet[:400]
        elif smell_type == "LongMethod" or smell_type == "DeepNesting":
            # Limit to a reasonable length for LongMethod and DeepNesting
            if len(snippet) > 50:
                snippet = snippet[:50]
        else:  # NoneSmellorUnknown
            snippet = snippet[:50]

        full_code = "\n".join(snippet)[:CONFIG["limits"]["snippet_max_length"]]
        if not full_code.strip():
            # If the snippet is empty, use a minimal fallback to avoid skipping the entry
            logger.warning(f"Extracted empty snippet for {smell_type} at {start_line}-{end_line}, using minimal fallback")
            full_code = "\n".join(lines[start_idx:end_idx][:1])  # Use the first line as a fallback
        logger.debug(f"Extracted snippet for {smell_type}: {len(full_code)} chars")
        return full_code
    except Exception as e:
        logger.error(f"Error extracting smell code for {smell_type}: {e}")
        # Fallback to a minimal snippet to avoid skipping
        return "\n".join(lines[start_idx:end_idx][:1]) or ""

def evaluate_smell(metrics: Dict, smell: str) -> Tuple[str, int]:
    for key in metrics:
        if not isinstance(metrics[key], (int, float)):
            metrics[key] = 0

    if smell == "LargeClass":
        loc = metrics.get("LOC", 0)
        nom = metrics.get("NOM", 0)
        if loc >= 400 and nom >= 15:
            severity = 3
        elif loc >= 300 and nom >= 12:
            severity = 2
        elif loc >= 200 and nom >= 10:
            severity = 1
        else:
            severity = 0
    elif smell == "LongMethod":
        loc = metrics.get("LOC_method", 0)
        cyclo = metrics.get("CYCLO_method", 0)
        if loc >= 100 and cyclo >= 10:
            severity = 3
        elif loc >= 75 and cyclo >= 8:
            severity = 2
        elif loc >= 50 and cyclo >= 6:
            severity = 1
        else:
            severity = 0
    elif smell == "LongParameterList":
        nop = metrics.get("NOP_method", 0)
        cyclo = metrics.get("CYCLO_method", 0)
        if nop >= 10 and cyclo >= 10:
            severity = 3
        elif nop >= 8 and cyclo >= 8:
            severity = 2
        elif nop >= 6 and cyclo >= 6:
            severity = 1
        else:
            severity = 0
    elif smell == "DeepNesting":
        nest = metrics.get("NEST", 0)
        if nest >= 8:
            severity = 3
        elif nest >= 6:
            severity = 2
        elif nest >= 4:
            severity = 1
        else:
            severity = 0
    else:
        return "NoneSmellorUnknown", 0

    logger.debug(f"Detected {smell} with severity {severity}, metrics: {metrics}")
    return smell, severity

def compute_metrics(file_path: Path) -> List[Dict]:
    start_time = time.time()
    results = []
    sample_count = 0

    file_size = os.path.getsize(file_path)
    if file_size > CONFIG["limits"]["max_file_size"]:
        logger.warning(f"Skipping large file {file_path}: {file_size} bytes")
        return results

    code_content = preprocess_code(file_path)
    if code_content is None:
        return results

    lizard_results = compute_lizard_metrics(file_path, code_content)
    if not lizard_results:
        logger.warning(f"No metrics computed for {file_path}")
        return results

    project = file_path.parts[len(EXTRACT_DIR.parts)]
    package = "/".join(file_path.parts[len(EXTRACT_DIR.parts):-1]) if len(file_path.parts) > len(
        EXTRACT_DIR.parts) + 1 else ""

    for method_name, metrics_full in lizard_results.items():
        try:
            if sample_count < 100:
                logger.debug(f"Sample {sample_count + 1} - {method_name}: {metrics_full}")
                sample_count += 1

            metrics_full["project"] = project
            detected_smells = []
            for smell in ["LargeClass", "LongMethod", "LongParameterList", "DeepNesting"]:
                smell_type, severity = evaluate_smell(metrics_full, smell)
                if severity > 0:
                    smell_code = extract_smell_code(code_content, metrics_full["start_line"], metrics_full["end_line"],
                                                    smell_type, metrics_full)
                    # Remove project field from metrics
                    cleaned_metrics = {k: v for k, v in metrics_full.items() if k != "project"}
                    results.append({
                        "smell_type": smell_type,
                        "severity": severity,
                        "project": project,
                        "package": package,
                        "method": method_name,
                        "smell_code": smell_code,
                        "metrics": cleaned_metrics,
                        "version": VERSION
                    })
                    detected_smells.append((smell_type, severity))

            if not detected_smells:
                smell_code = extract_smell_code(code_content, metrics_full["start_line"], metrics_full["end_line"],
                                                "NoneSmellorUnknown", metrics_full)
                cleaned_metrics = {k: v for k, v in metrics_full.items() if k != "project"}
                results.append({
                    "smell_type": "NoneSmellorUnknown",
                    "severity": 0,
                    "project": project,
                    "package": package,
                    "method": method_name,
                    "smell_code": smell_code,
                    "metrics": cleaned_metrics,
                    "version": VERSION
                })

        except Exception as e:
            logger.error(f"Error processing {method_name} in {file_path}: {e}")

    elapsed = time.time() - start_time
    if elapsed > 5:
        logger.warning(f"Slow processing for {file_path}: {elapsed:.2f} seconds")
    if results:
        logger.info(f"Successfully processed {file_path} with {len(results)} results")
    else:
        logger.error(f"Failed to process {file_path}: no results generated")
    return results

def write_to_json(results: List[Dict], first_write: bool = False):
    try:
        mode = 'w' if first_write else 'a'
        with open(OUTPUT_JSON, mode, encoding='utf-8') as jsonfile:
            for result in results:
                json.dump(result, jsonfile, ensure_ascii=False)
                jsonfile.write('\n')
        logger.debug(f"Wrote {len(results)} results to {OUTPUT_JSON}")
    except Exception as e:
        logger.error(f"Failed to write to JSON at {OUTPUT_JSON}: {e}")

def signal_handler(sig, frame):
    logger.info("Received interrupt signal, saving results and exiting...")
    global all_results
    if all_results:
        write_to_json(all_results)
    exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info(f"Starting analysis with {MAX_WORKERS} workers, version {VERSION}")

    if not EXTRACT_DIR.exists():
        logger.error(f"Extract directory {EXTRACT_DIR} does not exist")
        return

    all_files = [f for f in EXTRACT_DIR.rglob("*.py") if f.name != "Daily.py"]
    if not all_files:
        logger.error(f"No Python files found in {EXTRACT_DIR}")
        return

    logger.info(f"Found {len(all_files)} Python files to process")
    if os.path.exists(OUTPUT_JSON):
        os.remove(OUTPUT_JSON)

    global all_results
    all_results = []
    total_success, total_failure = 0, 0
    batch_size = CONFIG["limits"]["batch_size"]
    first_write = True

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_metrics, file_path): file_path for file_path in all_files}
        for i, future in enumerate(tqdm(futures, total=len(all_files), desc="Analyzing files")):
            file_path = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                total_success += len(results)
                if len(all_results) >= batch_size or (i + 1) == len(all_files):
                    write_to_json(all_results, first_write=first_write)
                    all_results = []
                    first_write = False
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                total_failure += 1

    if all_results:
        write_to_json(all_results, first_write=first_write)

    logger.info(f"Completed: Success: {total_success}, Failed: {total_failure}")

if __name__ == "__main__":
    main()