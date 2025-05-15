# MM-OPERA/evaluation/RIA/RIA_run.py

import os
import time
import requests
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

# Relative imports for modules within the 'evaluation' package
from ..config_loader import get_config, get_api_key, PROJECT_ROOT as CONFIG_PROJECT_ROOT
from ..utils import (
    setup_logger,
    save_json,
    load_json,
    encode_pil_image_to_base64,
    safe_dataset_iterator_with_error_info,
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
from datasets import config as hf_config

# Global logger, will be initialized in main
logger = None


def analyze_image_relationships_hf(
    model_name: str,
    dataset_ria_split,  # This is ds["ria"]
    config: dict,
    results_file_path: Path,
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

    model_identifier = model_config.get("model_identifier", model_name)
    provider_name = model_config.get("provider")
    provider_config = config["api_providers"].get(provider_name)
    if not provider_config:
        logger.error(f"Configuration for provider '{provider_name}' not found.")
        return {}

    try:
        api_key = get_api_key(model_name)
    except NameError:  # Fallback if get_api_key is not defined in this scope
        logger.error("'get_api_key' function not found. API key cannot be retrieved.")
        api_key = None  # Or fetch from config if available there
    if not api_key:
        logger.error(
            f"API key for model '{model_name}' could not be retrieved. Skipping."
        )
        return {}

    api_url = f"{provider_config.get('base_url', '').rstrip('/')}/{model_config.get('endpoint', '').lstrip('/')}"
    if not api_url:
        logger.error(
            f"Could not construct API URL for model '{model_name}'. Missing base_url or endpoint."
        )
        return {}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    ria_settings = config.get("evaluation_settings", {}).get("ria", {})
    prompt_template = ria_settings.get("prompt", "Default prompt if not found.")
    max_tokens = ria_settings.get("default_max_tokens", 4096)

    num_images_to_process = ria_settings.get("num_images_to_process", 2)
    sleep_time = ria_settings.get(
        "sleep_time_after_judge_api",
        config.get("general_settings", {}).get("sleep_time_between_requests", 1),
    )

    current_run_results_batch = {}  # Results for this specific run/batch

    processed_item_ids = set()
    try:
        all_results = load_json(results_file_path) or {}
        processed_item_ids = set(all_results.keys())
        logger.info(
            f"Loaded {len(processed_item_ids)} processed items from results file"
        )
    except Exception as e:
        logger.warning(f"Could not load existing results from {results_file_path}: {e}")
        all_results = {}

    original_total_items = len(dataset_ria_split)
    items_yielded_by_iterator = 0
    items_actually_processed_logic = 0  # Count of items that go through your main logic

    # --- Use the safe iterator ---
    # The safe_iterator will have its own inner tqdm bar (position=1)
    # The main_pbar will be the outer one (position=0)

    # Pass the logger to the safe iterator
    safe_gen = safe_dataset_iterator_with_error_info(
        dataset_ria_split,
        logger=logger,  # Pass your logger instance
        desc=f"RIA SafeIter {model_name}",
    )

    logger.info(
        f"Starting processing for {original_total_items} RIA items using safe iterator for model '{model_name}'."
    )

    with tqdm(
        total=original_total_items,
        desc=f"MainProc RIA {model_name}",
        position=0,
        mininterval=0.5,
    ) as main_pbar:

        # Use tqdm to display progress bar
        for idx, item in safe_gen:
            items_yielded_by_iterator += 1

            # item_id = item.get("foldername")
            item_id = item.get("id")
            if not item_id:
                item_id = f"ria-{idx}"
                logger.warning(
                    f"Item at original index {idx} (yielded by iterator) missing 'id'. Using generated ID: {item_id}"
                )

            if item_id in processed_item_ids:
                # logger.info(f"Skipping already processed item: {item_id}")
                main_pbar.update(1)  # Account for this item in the main progress
                continue

            # logger.info(f"Processing item: {item_id} (original index {idx})")

            pil_images = []
            for i_img in range(1, num_images_to_process + 1):
                img_key = f"image{i_img}"
                if img_key in item and item[img_key] is not None:
                    pil_images.append(item[img_key])
                else:
                    logger.warning(
                        f"Image '{img_key}' not found or is None for item '{item_id}' (original index {idx})."
                    )

            if len(pil_images) < num_images_to_process:
                logger.error(
                    f"Item '{item_id}' (original index {idx}) has fewer than {num_images_to_process} valid images. Required: {num_images_to_process}, Found: {len(pil_images)}. Skipping."
                )
                main_pbar.update(1)  # Account for this item
                continue

            encoded_images_data = []
            valid_images = True
            for i_enc, pil_img in enumerate(pil_images):
                # Make sure encode_pil_image_to_base64 is robust or wrapped in try-except
                try:
                    b64_string, mime_type = encode_pil_image_to_base64(pil_img)
                    if b64_string and mime_type:
                        encoded_images_data.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_string}"
                                },
                            }
                        )
                    else:
                        raise ValueError("Encoding returned None or empty values.")
                except Exception as enc_e:
                    logger.error(
                        f"Failed to encode image {i_enc+1} for item '{item_id}' (original index {idx}): {enc_e}. Skipping item."
                    )
                    valid_images = False
                    break

            if not valid_images:
                main_pbar.update(1)  # Account for this item
                continue

            if (
                len(encoded_images_data) < num_images_to_process
            ):  # Should be caught by pil_images check, but as a safeguard
                logger.error(
                    f"Item '{item_id}' (original index {idx}) resulted in fewer than {num_images_to_process} ENCODED images. Required: {num_images_to_process}, Found: {len(encoded_images_data)}. Skipping."
                )
                main_pbar.update(1)
                continue

            if num_images_to_process >= 2 and len(encoded_images_data) >= 2:
                user_content = [
                    encoded_images_data[0],
                    encoded_images_data[1],
                ]
            elif len(encoded_images_data) == 1:
                user_content = [encoded_images_data[0]]
            else:  # Should ideally not be reached if num_images_to_process > 0
                logger.error(
                    f"Insufficient encoded images for payload for item '{item_id}' (idx {idx}). Found {len(encoded_images_data)}, need at least 1 (or {num_images_to_process}). Skipping."
                )
                main_pbar.update(1)
                continue

            payload = {
                "model": model_identifier,  # Or the specific model string expected by the API if different
                "messages": [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
            }
            api_response_text_for_item = f"Error: API call not made or failed before content retrieval for item {item_id}"

            try:
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=60
                )  # Added timeout

                if response.status_code == 200:
                    api_result = response.json()
                    # Extract the response text (this can vary based on API provider)
                    response_text_content = (
                        api_result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    if response_text_content:
                        current_run_results_batch[item_id] = response_text_content
                        # logger.info(
                        #     f"Successfully processed item: {item_id} (original index {idx})"
                        # )
                        api_response_text_for_item = (
                            response_text_content  # For saving logic
                        )
                    else:
                        logger.warning(
                            f"Item '{item_id}' (original index {idx}) processed by API, but no content in response: {api_result}"
                        )
                        api_response_text_for_item = (
                            "Error: No content in API response."
                        )
                        # current_run_results_batch[item_id] = api_response_text_for_item

                    items_actually_processed_logic += 1

                else:
                    logger.error(
                        f"API Error for item {item_id} (original index {idx}): Status {response.status_code} - {response.text[:500]}"  # Log snippet of error
                    )
                    api_response_text_for_item = f"Error: API returned status {response.status_code}. Response: {response.text[:200]}"
                    # current_run_results_batch[item_id] = api_response_text_for_item

            except requests.exceptions.Timeout:
                logger.error(
                    f"Request timed out for item {item_id} (original index {idx}) using URL {api_url}"
                )
                api_response_text_for_item = "Error: Request timed out."
                # current_run_results_batch[item_id] = api_response_text_for_item
            except requests.exceptions.RequestException as e_req:
                logger.error(
                    f"Request failed for item {item_id} (original index {idx}): {e_req}"
                )
                api_response_text_for_item = f"Error: Request failed - {str(e_req)}"
                # current_run_results_batch[item_id] = api_response_text_for_item
            except (
                Exception
            ) as e_proc:  # Catch other errors during API call or response parsing
                logger.error(
                    f"Unexpected error during API interaction for item {item_id} (original index {idx}): {e_proc}"
                )
                api_response_text_for_item = (
                    f"Error: Unexpected during API interaction - {str(e_proc)}"
                )
                # current_run_results_batch[item_id] = api_response_text_for_item

            # --- Saving logic (after each item or batching) ---
            if item_id in current_run_results_batch:
                try:
                    all_results[item_id] = current_run_results_batch[
                        item_id
                    ]  # Update with the specific item's result
                    save_json(all_results, results_file_path)
                    processed_item_ids.add(item_id)  # 更新已处理项目集合
                except Exception as e_save:
                    logger.error(f"Error saving results for item {item_id}: {e_save}")

            if item_id in current_run_results_batch:  # Clear after saving
                del current_run_results_batch[item_id]

            main_pbar.update(1)  # Update main progress after processing this item
            time.sleep(sleep_time)

        items_skipped_by_iterator = original_total_items - items_yielded_by_iterator
        if items_skipped_by_iterator > 0:
            if (
                main_pbar.n < original_total_items
            ):  # If current progress is less than total
                main_pbar.update(
                    items_skipped_by_iterator
                )  # Add the count of skipped items
            main_pbar.set_postfix_str(
                f"{items_skipped_by_iterator} skipped by iterator; {items_actually_processed_logic} processed by logic",
                refresh=True,
            )
        else:
            main_pbar.set_postfix_str(
                f"{items_actually_processed_logic} processed by logic", refresh=True
            )

    # Final save of any remaining batch (should be empty if saving per item and clearing)
    if current_run_results_batch:  # This implies an issue if not empty
        logger.warning(
            f"Found {len(current_run_results_batch)} items in batch at the end. Saving them now."
        )
        try:
            all_results.update(current_run_results_batch)
            save_json(all_results, results_file_path)
        except Exception as e_final_save:
            logger.error(f"Error during final save of batched results: {e_final_save}")

    logger.info(f"RIA run processing for model '{model_name}' complete.")
    logger.info(f"Total items in dataset: {original_total_items}")
    logger.info(f"Items yielded by safe iterator: {items_yielded_by_iterator}")
    logger.info(f"Items processed by core logic: {items_actually_processed_logic}")


def main():
    global logger  # Allow main to set the global logger

    parser = argparse.ArgumentParser(
        description="Run RIA evaluation for a specified model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # required=True,
        help="Name of the model to evaluate (must be defined in model_config.yaml).",
    )
    args = parser.parse_args()
    model_name_to_run = args.model_name

    try:
        # --- 1. Load Configuration ---
        config = get_config()  # Uses default path: evaluation/model_config.yaml
        if not model_name_to_run:
            model_name_to_run = config["evaluation_settings"]["ria"][
                "default_model_name"
            ]

        # --- 2. Setup Paths ---
        # Output paths based on project root and config
        # Results will be like: MM-OPERA/results/RIA_gpt-4o_results.json
        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]

        # Ensure base directories exist
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Specific file paths for this run
        log_file_path = logs_dir / f"{model_name_to_run}_RIA_run.log"
        results_file_path = results_dir / f"{model_name_to_run}_RIA_results.json"

        # --- 3. Setup Logger ---
        logger = setup_logger(f"{model_name_to_run}_RIA_run", log_file_path)
        logger.info(f"--- Starting RIA evaluation for model: {model_name_to_run} ---")
        logger.info(f"Using project root: {PROJECT_ROOT}")
        logger.info(f"Results will be saved to: {results_file_path}")
        logger.info(f"Logs will be saved to: {log_file_path}")

        # --- 4. Load Hugging Face Dataset ---
        hf_cache_dir = (
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_dataset_name = config["general_settings"]["huggingface_dataset_name"]

        # Set the cache directory for Hugging Face datasets
        hf_config.HF_DATASETS_CACHE = str(hf_cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache dir exists
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        try:
            dataset = load_dataset(hf_dataset_name)
            logger.info(f"Dataset '{hf_dataset_name}' loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face dataset '{hf_dataset_name}': {e}"
            )
            return  # Exit if dataset cannot be loaded

        if "ria" not in dataset:
            logger.error(
                f"'ria' split not found in dataset '{hf_dataset_name}'. Available splits: {list(dataset.keys())}"
            )
            return
        dataset_ria_split = dataset["ria"]
        logger.info(f"Using 'ria' split with {len(dataset_ria_split)} items.")

        # --- 6. Run Analysis ---
        analyze_image_relationships_hf(
            model_name=model_name_to_run,
            dataset_ria_split=dataset_ria_split,
            config=config,
            results_file_path=results_file_path,
        )


        logger.info(f"Final results saved to {results_file_path}")
        logger.info(f"--- RIA evaluation finished for model: {model_name_to_run} ---")

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
    main()
