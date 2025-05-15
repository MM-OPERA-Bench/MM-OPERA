# MM-OPERA/evaluation/RIA/RIA_regular_judge.py

import os
import requests
import json
import time
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

try:
    from ..config_loader import get_config, get_api_key, PROJECT_ROOT
    from ..utils import setup_logger, save_json, load_json
except ImportError:
    print(
        "Failed to import from parent directory. Ensure script is run in a way that allows relative imports, or adjust sys.path."
    )
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config_loader import get_config, get_api_key, PROJECT_ROOT
    from utils import setup_logger, save_json, load_json

# Hugging Face datasets library
from datasets import load_dataset
from datasets import config as hf_config

# Global logger
logger = None

# =======================
# Helper Functions for Batching
# =======================


def generate_prompt_batch(batch_prompts_info):
    """
    Combines multiple prompts into a single batched prompt.

    Args:
        batch_prompts_info (list of dict): Each dict contains 'item_id' and 'user_prompt'.

    Returns:
        str: Combined user prompt for the batch.
        list of str: Mapping of item_ids for each prompt.
    """
    combined_prompt_text = ""
    mapping_info = []
    for idx, prompt_info_item in enumerate(batch_prompts_info, 1):
        combined_prompt_text += f"{idx}. {prompt_info_item['user_prompt']}\n\n"
        mapping_info.append(prompt_info_item["item_id"])
    return combined_prompt_text, mapping_info


def generate_payload_batch(
    combined_prompt_text,
    judge_model_identifier,
    judge_prompt_template,
    max_tokens_for_batch,
):
    """
    Generates a single payload for the batch of prompts.

    Args:
        combined_prompt_text (str): Combined user prompt for the batch.
        judge_model_identifier (str): The identifier for the judge model.
        judge_prompt_template (str): The system prompt for the judge.
        max_tokens_for_batch (int): Max tokens for the API response.

    Returns:
        dict: Payload for the API request.
    """
    payload = {
        "model": judge_model_identifier,
        "messages": [
            {"role": "system", "content": judge_prompt_template},
            {"role": "user", "content": combined_prompt_text},
        ],
        "max_tokens": max_tokens_for_batch,
    }
    return payload


def generate_individual_prompts(item_data_ground_truth, mllm_item_result):
    """
    Generates individual prompts for each item.

    Args:
        item_data_ground_truth (dict): Ground truth data for the item from Hugging Face dataset.
        mllm_item_result (str): MLLM response for this item (from RIA_run.py results).

    Returns:
        dict: Contains 'item_id' and 'user_prompt'.
    """
    global logger
    item_id = item_data_ground_truth.get(
        "id"
    )
    if not item_id:
        logger.warning("No 'id' specified in ground truth item. Skipping.")
        return None

    if (
        not mllm_item_result
        or mllm_item_result.startswith("Error")
        or mllm_item_result == "No response generated."
    ):
        logger.error(
            f"MLLM result for item {item_id} is invalid: {mllm_item_result}. Skipping prompt generation for this item."
        )
        return None

    try:
        # Extract image descriptions from dataset
        img_desc_list = [
            item_data_ground_truth.get(f"description{i}", "")
            for i in range(1, 3)  # RIA uses 2 images
        ]

        # Extract relation and explanation information
        relation_gt = item_data_ground_truth.get("relation", "")
        explanation_gt = item_data_ground_truth.get("explanation", "")

    except (KeyError, IndexError, TypeError) as e:
        logger.error(
            f"Error extracting ground truth details for item {item_id}: {e}. Check dataset structure. Skipping."
        )
        return None

    # Construct the standard answer format
    standard_answer = {
        "First image": img_desc_list[0] if len(img_desc_list) > 0 else "",
        "Second image": img_desc_list[1] if len(img_desc_list) > 1 else "",
        "Relation": relation_gt,
        "Explanation": explanation_gt,
    }

    user_prompt_text = (
        f"MLLM Output: {json.dumps(mllm_item_result, ensure_ascii=False)} \n"
        f"Standard Answer: {json.dumps(standard_answer, ensure_ascii=False)}"
    )

    return {
        "item_id": item_id,
        "user_prompt": user_prompt_text,
    }


def judge_model_results_main(
    ground_truth_data,
    mllm_results_data,
    config,
    judge_model_name_arg,
    judge_scores_output_file,
):
    """
    Main function to orchestrate the judging process.
    Loads existing judge results, prepares batches for unjudged items, calls API, and returns updated results.
    """
    global logger

    # Load existing judge results if the file exists
    if judge_scores_output_file.exists():
        judge_results = load_json(judge_scores_output_file) or {}
        logger.info(
            f"Loaded {len(judge_results)} existing judge results from {judge_scores_output_file}"
        )
    else:
        judge_results = {}
        logger.info(
            f"No existing judge results found at {judge_scores_output_file}. Starting fresh."
        )

    ria_judge_config = config["evaluation_settings"]["ria"]["regular_judge"]
    general_config = config["general_settings"]

    if not judge_model_name_arg:
        judge_model_name_arg = ria_judge_config.get("default_judge_model_name")
        if not judge_model_name_arg:
            raise ValueError(
                "Judge model name not provided via argument and not found in config's default_judge_model_name."
            )
    judge_model_name = judge_model_name_arg
    judge_model_specific_config = config["models"].get(judge_model_name)
    if not judge_model_specific_config:
        logger.error(
            f"Judge model '{judge_model_name}' not found in model_config.yaml models section."
        )
        return judge_results

    judge_model_provider_name = judge_model_specific_config.get("provider")
    judge_api_provider_config = config["api_providers"].get(judge_model_provider_name)
    if not judge_api_provider_config:
        logger.error(
            f"API provider '{judge_model_provider_name}' for judge model '{judge_model_name}' not found."
        )
        return judge_results

    judge_base_url = judge_api_provider_config["base_url"]
    judge_endpoint = judge_model_specific_config.get("endpoint", "/chat/completions")
    judge_api_url = f"{judge_base_url.rstrip('/')}/{judge_endpoint.lstrip('/')}"

    judge_model_identifier = judge_model_specific_config.get(
        "model_identifier", judge_model_name
    )
    judge_api_key = get_api_key(judge_model_name)
    if not judge_api_key:
        logger.error(
            f"API key for judge model '{judge_model_name}' could not be retrieved."
        )
        return judge_results

    judge_headers = {
        "Authorization": f"Bearer {judge_api_key}",
        "Content-Type": "application/json",
    }

    judge_prompt_template = ria_judge_config["prompt"]
    judge_group_size = ria_judge_config.get("judge_group_size", 1)

    # Calculate max tokens for judge response
    default_max_tokens_per_item = ria_judge_config.get(
        "default_max_tokens_per_item_judge", 400
    )
    default_max_tokens_judge = ria_judge_config.get(
        "default_max_tokens_judge", default_max_tokens_per_item * judge_group_size
    )

    sleep_time = ria_judge_config.get(
        "sleep_time_after_judge_api"
    ) or general_config.get("sleep_time_between_judge_requests", 7)

    # Prepare batches of prompts for unjudged items
    all_prompts = []
    for item_id, mllm_result in mllm_results_data.items():
        if item_id in judge_results and "score_judge" in judge_results[item_id]:
            # logger.info(f"Item {item_id} already judged. Skipping.")
            continue

        # Find matching ground truth item
        matching_gt_item = None
        for gt_item in ground_truth_data:
            if gt_item.get("id") == item_id:
                matching_gt_item = gt_item
                break

        if not matching_gt_item:
            logger.warning(f"No ground truth data found for item {item_id}. Skipping.")
            continue

        prompt_info = generate_individual_prompts(matching_gt_item, mllm_result)
        if prompt_info:
            all_prompts.append(prompt_info)

    logger.info(f"Prepared {len(all_prompts)} prompts for judging.")

    if not all_prompts:
        logger.info("No new items to judge. Exiting.")
        return judge_results

    # Calculate total number of batches
    total_batches = (len(all_prompts) + judge_group_size - 1) // judge_group_size

    # Process in batches with progress bar
    for batch_start in tqdm(
        range(0, len(all_prompts), judge_group_size),
        total=total_batches,
        desc="Processing judge batches",
    ):
        batch_end = min(batch_start + judge_group_size, len(all_prompts))
        current_batch = all_prompts[batch_start:batch_end]

        # logger.info(
        #     f"Processing batch {batch_start//judge_group_size + 1}/{total_batches}"
        # )

        combined_prompt, item_id_mapping = generate_prompt_batch(current_batch)
        payload = generate_payload_batch(
            combined_prompt,
            judge_model_identifier,
            judge_prompt_template,
            default_max_tokens_judge,
        )

        try:
            response = requests.post(
                judge_api_url, headers=judge_headers, json=payload, timeout=240
            )
            response.raise_for_status()

            result = response.json()
            response_text = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            if not response_text:
                logger.error("Empty response from judge API.")
                continue

            # Parse the JSON response
            try:
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                parsed_response = json.loads(response_text.strip())

                # Map the responses back to the original items
                for idx, item_id in enumerate(item_id_mapping, 1):
                    idx_str = str(idx)
                    if idx_str in parsed_response:
                        judge_results[item_id] = parsed_response[idx_str]
                        # logger.info(
                        #     f"Successfully judged item {item_id}: {parsed_response[idx_str]}"
                        # )
                    else:
                        logger.warning(
                            f"No judge result for item {item_id} in response."
                        )

                save_json(judge_results, judge_scores_output_file)
                # logger.info(
                #     f"Intermediate judge scores saved to {judge_scores_output_file} after batch {batch_start//judge_group_size + 1}/{total_batches}"
                # )

            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse judge response as JSON: {e}\nResponse: {response_text}"
                )
                continue

            # Sleep to avoid rate limits
            # logger.info(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            continue

    return judge_results


def combine_results_and_judgements(mllm_results_data, final_judge_scores):
    """
    Combines MLLM results with their corresponding judge scores.
    For RIA, each item has a single MLLM response and a single judge score.
    """
    global logger
    combined_output = {}
    for item_id, mllm_response in mllm_results_data.items():
        if item_id not in final_judge_scores:
            logger.warning(
                f"Judge scores not found for item_id: {item_id} during combination."
            )
            combined_output[item_id] = [
                {
                    "MLLM_answer": mllm_response,
                    "judge_answer": {
                        "score_judge": "N/A",
                        "score_reason": "JUDGEMENT_DATA_MISSING",
                    },
                }
            ]
            continue

        judge_score = final_judge_scores[item_id]
        combined_output[item_id] = [
            {
                "MLLM_answer": mllm_response,
                "judge_answer": judge_score,
            }
        ]
    return combined_output


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Run RIA judging for a specified test model's results."
    )
    parser.add_argument(
        "--test_model_name",
        type=str,
        # required=True,
        help="Name of the MLLM whose results are to be judged (must match how its results file is named, e.g., 'Gemini-2.0-Flash-Thinking-Exp').",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default=None,
        help="Name of the LLM to use as the judge (e.g., GPT-4o-judge). Must be defined in model_config.yaml.",
    )
    args = parser.parse_args()
    temp_logger_for_config_errors = None

    try:
        config = get_config()
        test_model_to_judge = args.test_model_name
        if not test_model_to_judge:
            test_model_to_judge = config["evaluation_settings"]["ria"][
                "default_model_name"
            ]

        results_base_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_base_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]

        results_base_dir.mkdir(parents=True, exist_ok=True)
        logs_base_dir.mkdir(parents=True, exist_ok=True)

        mllm_results_file = results_base_dir / f"{test_model_to_judge}_RIA_results.json"

        judge_log_file = logs_base_dir / f"{test_model_to_judge}_RIA_regular_judge.log"
        judge_scores_output_file = (
            results_base_dir / f"{test_model_to_judge}_RIA_regular_scoring.json"
        )
        # combined_output_file = (
        #     results_base_dir / f"{test_model_to_judge}_RIA_regular_combined_results.json"
        # )

        logger = setup_logger(
            f"{test_model_to_judge}_RIA_regular_judge", judge_log_file
        )
        temp_logger_for_config_errors = logger

        logger.info(f"--- Starting RIA Judging for MLLM: {test_model_to_judge} ---")
        logger.info(f"Using Project Root: {PROJECT_ROOT}")
        logger.info(f"Reading MLLM results from: {mllm_results_file}")
        logger.info(f"Saving judge scores to: {judge_scores_output_file}")
        # logger.info(f"Saving combined results to: {combined_output_file}")
        logger.info(f"Logging to: {judge_log_file}")

        config["general_settings"]["current_judge_file_path"] = str(
            judge_scores_output_file
        )

        if not mllm_results_file.exists():
            logger.error(f"MLLM results file not found: {mllm_results_file}")
            return

        # Set up Hugging Face cache directory
        hf_cache_dir_path = (
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_cache_dir_path.mkdir(parents=True, exist_ok=True)
        hf_config.HF_DATASETS_CACHE = str(hf_cache_dir_path)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir_path)
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        # Load dataset from Hugging Face
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

        if "ria" not in dataset:
            logger.error(
                f"'ria' split not found in dataset '{hf_dataset_name}'. Available splits: {list(dataset.keys())}"
            )
            return

        # Remove image columns to avoid loading images
        image_columns = [
            col for col in dataset["ria"].column_names if col.startswith("image")
        ]
        dataset_ria_split = dataset["ria"].remove_columns(image_columns)
        # logger.info(f"Removed image columns: {image_columns}")
        # logger.info(f"Remaining columns: {dataset_ria_split.column_names}")

        # Convert dataset to list of dictionaries
        ground_truth_data_loaded = [item for item in dataset_ria_split]
        logger.info(f"Loaded {len(ground_truth_data_loaded)} items from dataset.")

        mllm_results_data_loaded = load_json(mllm_results_file)
        if not mllm_results_data_loaded:
            logger.error(
                f"Failed to load or empty MLLM results from {mllm_results_file}"
            )
            return

        final_judge_scores_data = judge_model_results_main(
            ground_truth_data_loaded,
            mllm_results_data_loaded,
            config,
            args.judge_model_name,
            judge_scores_output_file,
        )

        # Save updated judge results
        save_json(final_judge_scores_data, judge_scores_output_file)
        logger.info(
            f"All judge scores finalized and saved to {judge_scores_output_file}"
        )

        # combined_data_final = combine_results_and_judgements(
        #     mllm_results_data_loaded, final_judge_scores_data
        # )

        # save_json(combined_data_final, combined_output_file)
        # logger.info(
        #     f"Combined MLLM results and judge scores saved to {combined_output_file}"
        # )

        logger.info(f"--- RIA Judging finished for MLLM: {test_model_to_judge} ---")
    except FileNotFoundError as fnf_e:
        log_func = (
            temp_logger_for_config_errors.error
            if temp_logger_for_config_errors
            else print
        )
        if temp_logger_for_config_errors:
            log_func(
                f"Configuration or essential file not found: {fnf_e}", exc_info=True
            )
        else:
            log_func(f"Configuration or essential file not found: {fnf_e}")
    except yaml.YAMLError as yaml_e:
        log_func = (
            temp_logger_for_config_errors.error
            if temp_logger_for_config_errors
            else print
        )
        if temp_logger_for_config_errors:
            log_func(f"YAML parsing error in configuration: {yaml_e}", exc_info=True)
        else:
            log_func(f"YAML parsing error in configuration: {yaml_e}")
    except KeyError as key_e:
        log_func = (
            temp_logger_for_config_errors.error
            if temp_logger_for_config_errors
            else print
        )
        if temp_logger_for_config_errors:
            log_func(f"Missing key in configuration file: {key_e}", exc_info=True)
        else:
            log_func(f"Missing key in configuration file: {key_e}")
    except Exception as e:
        log_func_error = (
            logger.error
            if logger
            else (
                temp_logger_for_config_errors.error
                if temp_logger_for_config_errors
                else print
            )
        )
        log_func_error(
            f"An unexpected error occurred in main_judge: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
