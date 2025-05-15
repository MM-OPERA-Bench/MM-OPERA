# MM-OPERA/evaluation/ICA/ICA_run.py

import time
import requests
import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm

# Relative imports for modules within the 'evaluation' package
from ..config_loader import get_config, get_api_key, PROJECT_ROOT as CONFIG_PROJECT_ROOT
from ..utils import (
    setup_logger,
    save_json,
    load_json,
    encode_pil_image_to_base64,
    PROJECT_ROOT as UTILS_PROJECT_ROOT,
    safe_dataset_iterator_with_error_info,
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


# =======================
# NEW FUNCTION TO BE IMPLEMENTED
# =======================
def analyze_image_relationships_hf(
    model_name,
    dataset_ica_split,
    config,
    results_file_path,
):
    global logger  # Use the global logger initialized in main

    logger.info(
        f"Starting ICA analysis with Hugging Face dataset for model: {model_name}"
    )

    # --- 1. Extract Configurations ---
    try:
        ica_config = config["evaluation_settings"]["ica"]
        prompt_template = ica_config["prompt"]
        default_max_tokens = ica_config.get(
            "default_max_tokens", 3000
        )  # Use .get for safety
        num_images_to_process_from_config = ica_config["num_images_to_process"]
        rotate_test = ica_config.get(
            "rotate_test", True
        )  # Default to True if not specified

        model_specific_config = config["models"].get(model_name)
        if not model_specific_config:
            logger.error(f"Model '{model_name}' not found in model_config.yaml.")
            return

        api_provider_name = model_specific_config["provider"]
        api_provider_config = config["api_providers"].get(api_provider_name)
        if not api_provider_config:
            logger.error(
                f"API provider '{api_provider_name}' for model '{model_name}' not found in model_config.yaml."
            )
            return

        base_url = api_provider_config["base_url"]
        endpoint = model_specific_config["endpoint"]
        api_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        api_key = get_api_key(
            model_name
        )  # Pass model_name instead of api_provider_name and config
        if not api_key:
            logger.error(
                f"API key for model '{model_name}' (provider '{api_provider_name}') could not be retrieved."
            )
            return

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        sleep_time = config.get("general_settings", {}).get(
            "sleep_time_between_requests", 1
        )

    except KeyError as e:
        logger.error(f"Missing configuration key: {e}. Please check model_config.yaml.")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    if num_images_to_process_from_config != 3:
        logger.warning(
            f"ICA task expects num_images_to_process to be 3 in config, but found {num_images_to_process_from_config}. "
            f"The ICA logic will use 3 images per sub-test as per its design."
        )

    # --- 2. Load Existing Results (if any) ---
    all_results = load_json(results_file_path) or {}
    logger.info(f"Loaded {len(all_results)} existing results from {results_file_path}")

    # 从结果文件中获取已处理的项目ID
    processed_item_ids = set(all_results.keys())
    logger.info(f"Found {len(processed_item_ids)} processed item IDs in results file")

    # --- 3. Iterate Through Dataset ---
    total_items = len(dataset_ica_split)
    logger.info(f"Processing {total_items} items from the ICA dataset split.")

    # 使用安全的数据集迭代器
    safe_gen = safe_dataset_iterator_with_error_info(
        dataset_ica_split, logger=logger, desc=f"ICA SafeIter {model_name}"
    )

    with tqdm(
        total=total_items,
        desc=f"MainProc ICA {model_name}",
        position=0,
        mininterval=0.5,
    ) as main_pbar:
        for idx, item in safe_gen:
            item_id = item.get("id")  #  or item.get("foldername")
            if not item_id:
                item_id = f"ica-{idx}"
                logger.warning(
                    f"Item at original index {idx} missing 'id'. Using generated ID: {item_id}"
                )

            item_id_str = str(item_id)  # Ensure item_id is a string for JSON keys

            # Initial check if the entire item was marked as processed.
            # We will still check sub-tests for errors later.
            if item_id_str in processed_item_ids and item_id_str in all_results:
                # If all sub-tests for this item have valid (non-Error) results, we can truly skip.
                # Otherwise, we need to enter the loop to re-process erroneous sub-tests.
                item_data_in_results = all_results.get(item_id_str, [])
                if isinstance(item_data_in_results, list) and item_data_in_results:
                    all_sub_tests_valid = True
                    for res_entry in item_data_in_results:
                        if isinstance(res_entry, str) and "Error" in res_entry:
                            all_sub_tests_valid = False
                            logger.info(
                                f"Item {item_id_str} (index {idx+1}/{total_items}) was processed but contains an error. Will re-check sub-tests."
                            )
                            break
                    if all_sub_tests_valid:
                        logger.info(
                            f"Item {item_id_str} (index {idx+1}/{total_items}) already processed with valid results. Skipping."
                        )
                        main_pbar.update(1)  # 更新主进度条
                        continue
                elif (
                    not item_data_in_results
                ):  # No results for this item_id, should process
                    logger.info(
                        f"Item {item_id_str} (index {idx+1}/{total_items}) in progress file but no results found. Will process."
                    )
                    pass  # Continue to processing
                # else: continue if item_id_str in processed_item_ids and not in all_results (should not happen with good progress saving)

            logger.info(
                f"Processing item {item_id_str} (index {idx+1}/{total_items})..."
            )

            pil_images_from_dataset = []
            for i in range(1, 5):
                img_data = item.get(f"image{i}")
                if img_data is not None:
                    pil_images_from_dataset.append(img_data)

            pil_images = pil_images_from_dataset
            num_available_images = len(pil_images)

            required_images_for_item = 4 if rotate_test else 3

            if num_available_images < required_images_for_item:
                logger.error(
                    f"Item {item_id_str} has {num_available_images} images, but {required_images_for_item} are required for current ICA settings (rotate_test={rotate_test}). Skipping."
                )
                all_results[item_id_str] = [
                    f"Error: Insufficient images ({num_available_images} provided, {required_images_for_item} required)"
                ] * (4 if rotate_test else 1)
                save_json(all_results, results_file_path)
                main_pbar.update(1)  # 更新主进度条
                continue

            encoded_images_data = []
            try:
                images_to_encode = pil_images[:required_images_for_item]
                for i, pil_image in enumerate(images_to_encode):
                    if pil_image is None:
                        logger.error(
                            f"Image at index {i} for item {item_id_str} is None. Skipping item."
                        )
                        raise ValueError(f"Image at index {i} is None.")

                    base64_image, mime_type = encode_pil_image_to_base64(pil_image)
                    encoded_images_data.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error encoding images for item {item_id_str}: {e}. Skipping."
                )
                num_sub_tests_for_error = 4 if rotate_test else 1
                all_results[item_id_str] = [
                    f"Error: Image encoding failed - {e}"
                ] * num_sub_tests_for_error
                save_json(all_results, results_file_path)
                main_pbar.update(1)  # 更新主进度条
                continue

            item_responses = all_results.get(item_id_str, [])
            if not isinstance(
                item_responses, list
            ):  # Ensure it's a list for sub-test results
                item_responses = []

            if rotate_test:
                image_index_sets = [[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]]
                num_tests = 4
            else:
                image_index_sets = [[0, 1, 2]]
                num_tests = 1

            # Ensure item_responses has the correct number of slots
            if len(item_responses) < num_tests:
                item_responses.extend(
                    ["Not Processed Yet"] * (num_tests - len(item_responses))
                )
            elif (
                len(item_responses) > num_tests
            ):  # Should not happen if config is consistent
                logger.warning(
                    f"Item {item_id_str} has {len(item_responses)} results, but expected {num_tests}. Truncating."
                )
                item_responses = item_responses[:num_tests]

            item_fully_processed_without_error = (
                True  # Flag to track if all sub-tests for this item are good
            )

            for test_idx in range(num_tests):
                current_response = item_responses[test_idx]

                # MODIFIED LOGIC: Check if this specific sub-test already has a valid-looking result
                # A result is NOT satisfactory if it contains "Error"
                is_processed_satisfactorily = (
                    isinstance(current_response, str)
                    and "Error"
                    not in current_response  # Key change: re-process if "Error" is in the string
                    and current_response
                    not in ["No response generated.", "Not Processed Yet", None]
                    and len(current_response) > 10  # Crude check for meaningful content
                )

                if is_processed_satisfactorily:
                    logger.info(
                        f"Sub-test {test_idx+1}/{num_tests} for item {item_id_str} already has a valid result. Skipping."
                    )
                    continue
                else:
                    if (
                        isinstance(current_response, str)
                        and "Error" in current_response
                    ):
                        logger.info(
                            f"Sub-test {test_idx+1}/{num_tests} for item {item_id_str} had an error: '{current_response[:100]}...'. Re-processing."
                        )
                    elif (
                        current_response != "Not Processed Yet"
                    ):  # Log if it was some other unsatisfactory state
                        logger.info(
                            f"Sub-test {test_idx+1}/{num_tests} for item {item_id_str} was not satisfactorily processed ('{str(current_response)[:50]}...'). Re-processing."
                        )
                    # "Not Processed Yet" is the default, so no special log for that, just proceed.

                item_fully_processed_without_error = (
                    False  # If we are here, at least one sub-test needs processing
                )

                current_image_indices = image_index_sets[test_idx]
                current_images_content = []
                try:
                    for img_idx in current_image_indices:
                        current_images_content.append(encoded_images_data[img_idx])
                except IndexError:
                    logger.error(
                        f"Error selecting images for item {item_id_str}, test {test_idx+1}. Image index {img_idx} out of bounds for {len(encoded_images_data)} encoded images. Skipping test."
                    )
                    item_responses[test_idx] = (
                        "Error: Image selection failed (index out of bounds)"
                    )
                    save_json(all_results, results_file_path)  # Save error immediately
                    continue  # Move to next sub-test

                payload = {
                    "model": model_specific_config.get("model_identifier", model_name),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant specialized in image analysis.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_template},
                                *current_images_content,
                            ],
                        },
                    ],
                    "max_tokens": model_specific_config.get(
                        "max_tokens", default_max_tokens
                    ),
                }

                for param_key, param_value in model_specific_config.items():
                    if param_key not in [
                        "provider",
                        "endpoint",
                        "max_tokens",
                        "model_identifier",
                    ]:
                        payload[param_key] = param_value

                logger.debug(
                    f"Sending payload for item {item_id_str}, test {test_idx+1}: {json.dumps(payload, indent=2)}"
                )

                try:
                    response = requests.post(
                        api_url, headers=headers, json=payload, timeout=120
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    logger.debug(
                        f"Received response for item {item_id_str}, test {test_idx+1}: {json.dumps(response_json, indent=2)}"
                    )

                    if (
                        response_json.get("choices")
                        and len(response_json["choices"]) > 0
                    ):
                        message = response_json["choices"][0].get("message", {})
                        content = message.get("content", "No response generated.")
                        item_responses[test_idx] = content
                    elif response_json.get("error"):
                        error_msg = response_json["error"].get(
                            "message", "Unknown API error"
                        )
                        logger.error(
                            f"API error for item {item_id_str}, test {test_idx+1}: {error_msg}"
                        )
                        item_responses[test_idx] = f"Error: API Error - {error_msg}"
                    else:
                        logger.warning(
                            f"Unexpected response structure for item {item_id_str}, test {test_idx+1}. Full response: {response_json}"
                        )
                        item_responses[test_idx] = (
                            "Error: Unexpected response structure"
                        )

                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Request failed for item {item_id_str}, test {test_idx+1}: {e}"
                    )
                    item_responses[test_idx] = f"Error: Request failed - {e}"
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during API call for item {item_id_str}, test {test_idx+1}: {e}",
                        exc_info=True,
                    )
                    item_responses[test_idx] = f"Error: Unexpected - {e}"

                all_results[item_id_str] = item_responses  # Update results for the item
                save_json(
                    all_results, results_file_path
                )  # Save after each sub-test API call

                logger.info(
                    f"Completed API call for item {item_id_str}, test {test_idx+1}/{num_tests}. Result snippet: {str(item_responses[test_idx])[:100]}..."
                )

                # Check if the newly obtained response still contains "Error"
                if (
                    isinstance(item_responses[test_idx], str)
                    and "Error" in item_responses[test_idx]
                ):
                    item_fully_processed_without_error = (
                        False  # Mark that this item still has errors
                    )

                if test_idx < num_tests - 1:  # If not the last sub-test for this item
                    logger.debug(
                        f"Sleeping for {sleep_time} seconds before next API call for this item..."
                    )
                    time.sleep(sleep_time)

            # After all sub-tests for an item are attempted
            # Update item_fully_processed_without_error based on the final state of item_responses
            item_fully_processed_without_error = True
            for res in item_responses:
                if isinstance(res, str) and "Error" in res:
                    item_fully_processed_without_error = False
                    break

            if item_fully_processed_without_error:
                logger.info(
                    f"Item {item_id_str} successfully processed for all sub-tests."
                )
            else:
                # If the item is already in the results file but still has errors, record a warning
                if item_id_str in processed_item_ids:
                    logger.warning(
                        f"Item {item_id_str} has errors after processing. It may be re-attempted if script is run again."
                    )

            all_results[item_id_str] = item_responses
            save_json(all_results, results_file_path)

            # Update the processed item IDs set
            processed_item_ids = set(all_results.keys())

            logger.info(
                f"Saved results for item {item_id_str}. Total items in results file: {len(processed_item_ids)}/{total_items}"
            )

            main_pbar.update(1)  # 更新主进度条

            if idx < total_items - 1:  # If not the last item overall
                logger.debug(
                    f"Sleeping for {sleep_time} seconds before processing next item..."
                )
                time.sleep(sleep_time)  # Sleep between items as well

    logger.info("ICA analysis with Hugging Face dataset finished.")
    return all_results


# =======================


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Run ICA evaluation for a specified model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # required=True,
        help="Name of the model to evaluate (must be defined in model_config.yaml).",
    )
    args = parser.parse_args()
    model_name_to_run = args.model_name

    _logger_for_early_errors = None
    try:
        config = get_config()
        if not model_name_to_run:
            model_name_to_run = config["evaluation_settings"]["ica"][
                "default_model_name"
            ]

        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]

        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = logs_dir / f"{model_name_to_run}_ICA_run.log"
        results_file_path = results_dir / f"{model_name_to_run}_ICA_results.json"

        # Initialize the global logger
        logger = setup_logger(f"{model_name_to_run}_ICA_run", log_file_path)
        _logger_for_early_errors = (
            logger  # Assign to the temp logger as well now that it's initialized
        )

        logger.info(f"--- Starting ICA evaluation for model: {model_name_to_run} ---")
        logger.info(f"Using project root: {PROJECT_ROOT}")
        logger.info(f"Results will be saved to: {results_file_path}")
        logger.info(f"Logs will be saved to: {log_file_path}")

        hf_cache_dir_path = (
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_cache_dir_path.mkdir(parents=True, exist_ok=True)
        hf_config.HF_DATASETS_CACHE = str(hf_cache_dir_path)
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        hf_dataset_name = config["general_settings"]["huggingface_dataset_name"]

        try:
            dataset = load_dataset(hf_dataset_name)
            logger.info(f"Dataset '{hf_dataset_name}' loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face dataset '{hf_dataset_name}' from cache '{hf_config.HF_DATASETS_CACHE}': {e}",
                exc_info=True,
            )
            return

        if "ica" not in dataset:
            logger.error(
                f"'ica' split not found in dataset '{hf_dataset_name}'. Available splits: {list(dataset.keys())}"
            )
            return
        dataset_ica_split = dataset["ica"]
        logger.info(f"Using 'ica' split with {len(dataset_ica_split)} items.")

        analyze_image_relationships_hf(
            model_name=model_name_to_run,
            dataset_ica_split=dataset_ica_split,
            config=config,
            results_file_path=results_file_path,
        )

        logger.info(f"ICA evaluation finished for model: {model_name_to_run}.")
        logger.info(f"Final results saved to {results_file_path}")

    except FileNotFoundError as e:
        # Use a basic print if logger failed to initialize, otherwise use the logger
        log_func = _logger_for_early_errors.error if _logger_for_early_errors else print
        log_func(
            f"Configuration or essential file not found: {e}",
            exc_info=True if _logger_for_early_errors else False,
        )
    except yaml.YAMLError as e:
        log_func = _logger_for_early_errors.error if _logger_for_early_errors else print
        log_func(
            f"YAML parsing error in configuration: {e}",
            exc_info=True if _logger_for_early_errors else False,
        )
    except Exception as e:
        log_func = _logger_for_early_errors.error if _logger_for_early_errors else print
        log_func(
            f"An unexpected error occurred in main: {e}",
            exc_info=True if _logger_for_early_errors else False,
        )


if __name__ == "__main__":
    main()
