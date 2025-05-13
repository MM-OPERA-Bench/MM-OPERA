# MM-OPERA/evaluation/RIA/RIA_run.py

import time
import requests
import argparse
from pathlib import Path
import yaml

# Relative imports for modules within the 'evaluation' package
from ..config_loader import get_config, get_api_key, PROJECT_ROOT as CONFIG_PROJECT_ROOT
from ..utils import (
    setup_logger,
    save_json,
    load_json,
    encode_pil_image_to_base64,
    PROJECT_ROOT as UTILS_PROJECT_ROOT,
)

# Ensure project roots from different modules are consistent (they should be)
if CONFIG_PROJECT_ROOT != UTILS_PROJECT_ROOT:
    raise ImportError(
        "Project root mismatch between config_loader and utils. Check paths."
    )
PROJECT_ROOT = UTILS_PROJECT_ROOT  # Use one consistent project root

# Hugging Face datasets library
from datasets import load_dataset
from datasets import config as hf_config  # To set cache directory

# Global logger, will be initialized in main
logger = None


def analyze_image_relationships_hf(
    model_name: str,
    dataset_ria_split,  # This is ds["ria"]
    config: dict,
    processed_item_ids: set,
    results_file_path: Path,
    progress_file_path: Path,
):
    """
    Analyzes image relationships using Hugging Face dataset format.
    """
    global logger
    if logger is None:  # Should be set by main
        print("Error: Logger not initialized in analyze_image_relationships_hf.")
        return {}

    model_config = config["models"].get(model_name)
    if not model_config:
        logger.error(f"Configuration for model '{model_name}' not found.")
        return {}

    provider_name = model_config["provider"]
    provider_config = config["api_providers"].get(provider_name)
    if not provider_config:
        logger.error(f"Configuration for provider '{provider_name}' not found.")
        return {}

    api_key = get_api_key(model_name)  # Fetches from env var or config
    if not api_key:
        logger.error(
            f"API key for model '{model_name}' could not be retrieved. Skipping."
        )
        return {}

    api_url = provider_config["base_url"] + model_config["endpoint"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    ria_settings = config["evaluation_settings"]["ria"]
    prompt_template = ria_settings["prompt"]
    max_tokens = model_config.get(
        "default_max_tokens", ria_settings.get("default_max_tokens", 300)
    )
    num_images_to_process = ria_settings.get("num_images_to_process", 2)
    sleep_time = config["general_settings"].get("sleep_time_between_requests", 1)

    current_run_results = {}  # Results for this specific run/batch

    for idx, item in enumerate(dataset_ria_split):
        item_id = item.get(
            "foldername"
        )  # Using 'foldername' as a unique ID for progress tracking
        if not item_id:
            # Fallback or generate a unique ID if 'foldername' isn't always present or unique enough
            item_id = f"ria_item_{idx}"
            logger.warning(
                f"Item at index {idx} missing 'foldername'. Using generated ID: {item_id}"
            )

        if item_id in processed_item_ids:
            logger.info(f"Skipping already processed item: {item_id}")
            continue

        logger.info(f"Processing item: {item_id}")

        pil_images = []
        # The Hugging Face dataset directly provides PIL.Image objects
        # Assumes images are named image1, image2, ... in the dataset features
        for i in range(1, num_images_to_process + 1):
            img_key = f"image{i}"
            if img_key in item and item[img_key] is not None:
                pil_images.append(item[img_key])
            else:
                logger.warning(
                    f"Image '{img_key}' not found or is None for item '{item_id}'."
                )

        if len(pil_images) < num_images_to_process:
            logger.error(
                f"Item '{item_id}' has fewer than {num_images_to_process} valid images. Required: {num_images_to_process}, Found: {len(pil_images)}. Skipping."
            )
            continue

        encoded_images_data = []
        valid_images = True
        for i, pil_img in enumerate(pil_images):
            b64_string, mime_type = encode_pil_image_to_base64(pil_img)
            if b64_string and mime_type:
                encoded_images_data.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_string}"},
                    }
                )
            else:
                logger.error(
                    f"Failed to encode image {i+1} for item '{item_id}'. Skipping item."
                )
                valid_images = False
                break

        if not valid_images:
            continue

        # Construct payload - adapt if your prompt or image order needs to change
        # The original code has images in reverse order in the payload (image_url[1] then image_url[0])
        # Assuming you want to maintain that:
        if len(encoded_images_data) >= 2:
            user_content = [
                encoded_images_data[1],  # Second image from dataset (item['image2'])
                encoded_images_data[0],  # First image from dataset (item['image1'])
            ]
        elif len(encoded_images_data) == 1:  # If only one image, adjust as needed
            user_content = [encoded_images_data[0]]
        else:  # Should not happen due to earlier checks
            logger.error(
                f"Not enough encoded images for item '{item_id}'. This should not happen."
            )
            continue

        payload = {
            "model": model_name,  # Or the specific model string expected by the API if different
            "messages": [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=60
            )  # Added timeout

            if response.status_code == 200:
                api_result = response.json()
                # Extract the response text (this can vary based on API provider)
                # For OpenAI-like APIs:
                response_text = (
                    api_result.get("choices", [{}])[0].get("message", {}).get("content")
                )
                if response_text:
                    current_run_results[item_id] = response_text
                    logger.info(f"Successfully processed item: {item_id}")
                else:
                    logger.warning(
                        f"Item '{item_id}' processed, but no content in response: {api_result}"
                    )
                    current_run_results[item_id] = "Error: No content in API response."

                processed_item_ids.add(item_id)

                # Save progress and incremental results
                save_json(list(processed_item_ids), progress_file_path)

                # For results, load existing, update, and save back
                # This matches the original behavior of appending to a single file.
                all_results = load_json(results_file_path) or {}
                all_results.update(current_run_results)
                save_json(all_results, results_file_path)
                current_run_results.clear()  # Clear for next potential save

            else:
                logger.error(
                    f"API Error for item {item_id}: Status {response.status_code} - {response.text}"
                )
                current_run_results[item_id] = (
                    f"Error: API returned status {response.status_code}. Response: {response.text}"
                )
                # Decide if you want to save errors to the main results file or a separate error log
                all_results = load_json(results_file_path) or {}
                all_results.update(current_run_results)
                save_json(all_results, results_file_path)
                current_run_results.clear()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for item {item_id}: {e}")
            current_run_results[item_id] = f"Error: Request failed - {str(e)}"
            all_results = load_json(results_file_path) or {}
            all_results.update(current_run_results)
            save_json(all_results, results_file_path)
            current_run_results.clear()

        except Exception as e:
            logger.error(f"Unexpected error processing item {item_id}: {e}")
            current_run_results[item_id] = f"Error: Unexpected - {str(e)}"
            # Potentially save this error to results as well
            all_results = load_json(results_file_path) or {}
            all_results.update(current_run_results)
            save_json(all_results, results_file_path)
            current_run_results.clear()

        time.sleep(sleep_time)

    # Final save of any remaining results (should be empty if saving incrementally)
    if current_run_results:
        all_results = load_json(results_file_path) or {}
        all_results.update(current_run_results)
        save_json(all_results, results_file_path)

    logger.info("RIA run processing complete.")


def main():
    global logger  # Allow main to set the global logger

    parser = argparse.ArgumentParser(
        description="Run RIA evaluation for a specified model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to evaluate (must be defined in model_config.yaml).",
    )
    # You can add more arguments, e.g., --dataset_subset, --config_file
    args = parser.parse_args()
    model_name_to_run = args.model_name

    try:
        # --- 1. Load Configuration ---
        config = get_config()  # Uses default path: evaluation/model_config.yaml

        # --- 2. Setup Paths ---
        # Output paths based on project root and config
        # Results will be like: MM-OPERA/results/RIA_gpt-4o_results.json
        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]
        progress_dir = PROJECT_ROOT / config["general_settings"]["progress_base_dir"]

        # Ensure base directories exist
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        progress_dir.mkdir(parents=True, exist_ok=True)

        # Specific file paths for this run
        log_file_path = logs_dir / f"RIA_{model_name_to_run}.log"
        results_file_path = results_dir / f"RIA_{model_name_to_run}_results.json"
        progress_file_path = progress_dir / f"RIA_{model_name_to_run}_progress.json"

        # --- 3. Setup Logger ---
        logger = setup_logger(f"RIA_run_{model_name_to_run}", log_file_path)
        logger.info(f"Starting RIA evaluation for model: {model_name_to_run}")
        logger.info(f"Using project root: {PROJECT_ROOT}")
        logger.info(f"Results will be saved to: {results_file_path}")
        logger.info(f"Logs will be saved to: {log_file_path}")
        logger.info(f"Progress will be saved to: {progress_file_path}")

        # --- 4. Load Hugging Face Dataset ---
        hf_cache_dir = (
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_dataset_name = config["general_settings"]["huggingface_dataset_name"]

        # Set the cache directory for Hugging Face datasets
        hf_config.HF_DATASETS_CACHE = str(hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache dir exists
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        try:
            # This is where you'd use your load_dataset.py if it provides a loading function.
            # For now, directly loading as per your example.
            # You might need `huggingface-cli login` if the dataset is private.
            dataset = load_dataset(hf_dataset_name)
            logger.info(f"Dataset '{hf_dataset_name}' loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face dataset '{hf_dataset_name}': {e}"
            )
            return  # Exit if dataset cannot be loaded

        # Assuming 'ria' is a split/configuration in your dataset
        if "ria" not in dataset:
            logger.error(
                f"'ria' split not found in dataset '{hf_dataset_name}'. Available splits: {list(dataset.keys())}"
            )
            return
        dataset_ria_split = dataset["ria"]
        logger.info(f"Using 'ria' split with {len(dataset_ria_split)} items.")

        # --- 5. Load Progress ---
        processed_items = set(load_json(progress_file_path) or [])
        if processed_items:
            logger.info(
                f"Loaded {len(processed_items)} processed item IDs from {progress_file_path}"
            )

        # --- 6. Run Analysis ---
        analyze_image_relationships_hf(
            model_name=model_name_to_run,
            dataset_ria_split=dataset_ria_split,
            config=config,
            processed_item_ids=processed_items,
            results_file_path=results_file_path,
            progress_file_path=progress_file_path,
        )

        logger.info(f"RIA evaluation finished for model: {model_name_to_run}.")
        logger.info(f"Final results saved to {results_file_path}")

    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found: {e}")
        else:
            print(f"File not found: {e}")
    except yaml.YAMLError as e:
        if logger:
            logger.error(f"YAML parsing error: {e}")
        else:
            print(f"YAML parsing error: {e}")
    except Exception as e:
        if logger:
            logger.critical(
                f"An unexpected error occurred in main: {e}", exc_info=True
            )  # exc_info for traceback
        else:
            print(f"An unexpected error occurred in main: {e}")


if __name__ == "__main__":
    # Before running, ensure:
    # 1. model_config.yaml is in MM-OPERA/evaluation/
    # 2. Environment variables for API keys (e.g., AIGPTX_API_KEY) are set.
    # To run: python -m evaluation.RIA.RIA_run --model_name gpt-4o
    # (Run from the MM-OPERA project root directory)
    main()
