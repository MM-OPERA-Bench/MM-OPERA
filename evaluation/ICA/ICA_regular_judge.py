# MM-OPERA/evaluation/ICA/ICA_regular_judge.py

import os
import requests
import json
import time
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm  # Import tqdm for progress bars

# Assuming config_loader.py and utils.py are in a directory one level above
# e.g., MM-OPERA/evaluation/
# and this script is in MM-OPERA/evaluation/ICA/
try:
    from ..config_loader import get_config, get_api_key, PROJECT_ROOT
    from ..utils import setup_logger, save_json, load_json
except ImportError:
    # Fallback for direct execution if paths are not set up for relative imports
    print(
        "Failed to import from parent directory. Ensure script is run in a way that allows relative imports, or adjust sys.path."
    )
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config_loader import get_config, get_api_key, PROJECT_ROOT
    from utils import setup_logger, save_json, load_json

# Hugging Face datasets library
from datasets import load_dataset
from datasets import config as hf_config  # To set cache directory


# Global logger, will be initialized in main
logger = None


# =======================
# Helper Functions for Batching
# =======================
def generate_prompt_batch(batch_prompts_info):
    """
    Combines multiple prompts into a single batched prompt.

    Args:
        batch_prompts_info (list of dict): Each dict contains 'item_id', 'judge_id', and 'user_prompt'.

    Returns:
        str: Combined user prompt for the batch.
        list of tuple: Mapping of (item_id, judge_id) for each prompt.
    """
    combined_prompt_text = ""
    mapping_info = []
    for idx, prompt_info_item in enumerate(batch_prompts_info, 1):
        combined_prompt_text += f"{idx}. {prompt_info_item['user_prompt']}\n\n"
        mapping_info.append((prompt_info_item["item_id"], prompt_info_item["judge_id"]))
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


def generate_individual_prompts(
    item_data_ground_truth,
    mllm_item_results,
    judge_id_list_to_process,
    config_rotate_test,
):
    """
    Generates individual prompts for each judge ID within an item.

    Args:
        item_data_ground_truth (dict): Ground truth data for the item from Hugging Face dataset.
        mllm_item_results (list): List of MLLM responses for this item (from ICA_run.py results).
        judge_id_list_to_process (list): List of judge IDs (sub-test indices) to process for this item.
        config_rotate_test (bool): Configuration for rotate_test.

    Returns:
        list of dict: Each dict contains 'item_id', 'judge_id', and 'user_prompt'.
    """
    global logger  # Ensure logger is accessible
    prompts_for_item = []
    item_id = item_data_ground_truth.get("id")
    if not item_id:
        tqdm.write(
            f"Warning: No 'id' (item_id) specified in ground truth item. Skipping."
        )
        return prompts_for_item

    if not isinstance(mllm_item_results, list):
        tqdm.write(
            f"Error: MLLM results for item {item_id} is not a list: {mllm_item_results}. Skipping prompt generation for this item."
        )
        return prompts_for_item

    # Check for errors in MLLM results
    for judge_id in judge_id_list_to_process:
        if judge_id >= len(mllm_item_results):
            continue

        # Check for error strings in ICA_run.py results
        if isinstance(mllm_item_results[judge_id], str) and mllm_item_results[
            judge_id
        ].startswith("Error:"):
            tqdm.write(
                f"Info: Skipping prompt generation for item {item_id}, judge_id {judge_id} due to error in MLLM result: {mllm_item_results[judge_id][:100]}..."
            )
            continue

    try:
        # Extract image descriptions from dataset
        img_desc_list = [
            item_data_ground_truth.get(f"description{i}", "") for i in range(1, 5)
        ]

        # Extract relation and reasoning information
        relation_gt = item_data_ground_truth.get("relation", "")

        # Extract reasoning information from the 'reasoning' field
        # The reasoning field could be a string containing JSON data or already a Python object
        reasoning_data = item_data_ground_truth.get("reasoning", [])

        if isinstance(reasoning_data, str):
            try:
                reasoning_data = json.loads(reasoning_data)
            except json.JSONDecodeError:
                tqdm.write(
                    f"Warning: Failed to parse reasoning JSON for item {item_id}. Using empty values."
                )
                reasoning_data = []

        if isinstance(reasoning_data, list) and len(reasoning_data) >= 2:
            exp1_gt = (
                reasoning_data[0].get("explanation", "")
                if isinstance(reasoning_data[0], dict)
                else ""
            )
            exp2_gt = (
                reasoning_data[1].get("explanation", "")
                if isinstance(reasoning_data[1], dict)
                else ""
            )
            path1_gt = (
                reasoning_data[0].get("path", "")
                if isinstance(reasoning_data[0], dict)
                else ""
            )
            path2_gt = (
                reasoning_data[1].get("path", "")
                if isinstance(reasoning_data[1], dict)
                else ""
            )
        else:
            # Fallback if reasoning data is not properly structured
            tqdm.write(
                f"Warning: Reasoning data for item {item_id} is not properly structured. Using empty values."
            )
            exp1_gt = ""
            exp2_gt = ""
            path1_gt = ""
            path2_gt = ""

    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        tqdm.write(
            f"Error: Error extracting ground truth details for item {item_id}: {e}. Check dataset structure. Skipping."
        )
        return prompts_for_item

    num_expected_mllm_results = 4 if config_rotate_test else 1
    if len(mllm_item_results) < num_expected_mllm_results:
        tqdm.write(
            f"Warning: Item {item_id} has insufficient MLLM results ({len(mllm_item_results)} found, {num_expected_mllm_results} expected based on rotate_test={config_rotate_test}). Some sub-tests may not be judgeable."
        )

    for judge_id in judge_id_list_to_process:
        if judge_id >= len(mllm_item_results):
            tqdm.write(
                f"Warning: Attempting to access MLLM result for judge_id {judge_id} but only {len(mllm_item_results)} results exist for item {item_id}. Skipping this sub-test."
            )
            continue

        mllm_output_text = mllm_item_results[judge_id]

        try:
            standard_answer_image_4_desc = (
                img_desc_list[3 - judge_id]
                if 0 <= 3 - judge_id < len(img_desc_list)
                else ""
            )
            current_exp1 = exp1_gt if judge_id < 2 else exp2_gt
            current_exp2 = exp2_gt if judge_id < 2 else exp1_gt
            current_path1 = path1_gt if judge_id < 2 else path2_gt
            current_path2 = path2_gt if judge_id < 2 else path1_gt
        except IndexError as e:
            tqdm.write(
                f"Error: Error constructing standard answer for item {item_id}, judge_id {judge_id} due to IndexError: {e}. This might indicate an issue with img_desc_list (expected 4 descriptions) or reasoning structure in dataset. Skipping sub-test."
            )
            continue

        user_prompt_text = (
            f"MLLM Output: {json.dumps(mllm_output_text, ensure_ascii=False)} \n"
            f'Standard Answer: {json.dumps({"Image 4": standard_answer_image_4_desc, "relation": relation_gt, "explanation": f"{current_exp1} Similarly, {current_exp2}", "reasoning path 1": current_path1, "reasoning path 2": current_path2}, ensure_ascii=False)}'
        )

        prompts_for_item.append(
            {
                "item_id": item_id,
                "judge_id": judge_id,
                "user_prompt": user_prompt_text,
            }
        )
    return prompts_for_item


def judge_model_results_main(
    ground_truth_data,
    mllm_results_data,
    config,
    judge_model_name_arg,
    judge_scores_output_file,
):
    """
    Main function to orchestrate the judging process.
    Loads existing results, prepares batches, calls API, and saves results.
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

    ica_judge_config = config["evaluation_settings"]["ica"]["regular_judge"]
    ica_run_config = config["evaluation_settings"]["ica"]
    general_config = config["general_settings"]

    if not judge_model_name_arg:
        judge_model_name_arg = ica_judge_config.get("default_judge_model_name")
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

    judge_prompt_template = ica_judge_config["prompt"]
    judge_group_size = ica_judge_config["judge_group_size"]
    max_tokens_per_item = ica_judge_config.get("default_max_tokens_per_item_judge", 400)
    max_tokens_for_batch_judge = ica_judge_config.get(
        "default_max_tokens_judge", max_tokens_per_item * judge_group_size
    )

    config_rotate_test = ica_run_config.get("rotate_test", True)
    sleep_time = ica_judge_config.get(
        "sleep_time_after_judge_api",
        general_config.get("sleep_time_between_judge_requests", 7),
    )

    all_prompts_to_judge_in_batches = []

    # Add tqdm to the loop iterating through ground truth data
    for gt_item in tqdm(ground_truth_data, desc="Processing items for judging"):
        item_id = gt_item.get("id")
        if not item_id:
            tqdm.write("Warning: Ground truth item missing 'id'. Skipping.")
            continue

        num_expected_tests = 4 if config_rotate_test else 1

        if item_id not in mllm_results_data:
            # tqdm.write(
            #     f"Warning: No MLLM results found for item_id: {item_id}. Marking all sub-tests as N/A for judging."
            # )
            # judge_results[item_id] = [
            #     {
            #         "score_judge": "N/A",
            #         "score_reason": f"Skipped judging: MLLM results missing for item {item_id}.",
            #     }
            # ] * num_expected_tests
            # save_json(
            #     judge_results, judge_scores_output_file
            # )
            continue

        mllm_responses_for_item = mllm_results_data[item_id]
        if not isinstance(mllm_responses_for_item, list):
            tqdm.write(
                f"Error: MLLM results for item {item_id} is not a list: {mllm_responses_for_item}. Marking as error for judging."
            )
            # judge_results[item_id] = [
            #     {
            #         "score_judge": "N/A",
            #         "score_reason": f"Skipped judging: MLLM results for item {item_id} are malformed.",
            #     }
            # ] * num_expected_tests
            # save_json(
            #     judge_results, judge_scores_output_file
            # )
            continue

        if (
            item_id not in judge_results
            or not isinstance(judge_results.get(item_id), list)
            or len(judge_results.get(item_id)) != num_expected_tests
        ):
            tqdm.write(
                f"Debug: Initializing/Resetting judge scores structure for item {item_id} to {num_expected_tests} slots."
            )
            judge_results[item_id] = [None] * num_expected_tests

        current_item_judge_outputs = judge_results[item_id]
        judge_ids_to_process_for_this_item = []

        for i in range(num_expected_tests):
            if i >= len(mllm_responses_for_item):
                tqdm.write(
                    f"Warning: MLLM output for item {item_id}, sub-test {i} is missing. Expected {num_expected_tests} results, got {len(mllm_responses_for_item)}."
                )
                if i < len(current_item_judge_outputs):
                    pass
                    current_item_judge_outputs[i] = {
                        "score_judge": "N/A",
                        "score_reason": f"Skipped judging: MLLM output missing for sub-test {i}.",
                    }
                continue

            mllm_sub_test_output = mllm_responses_for_item[i]

            is_error = False
            error_reason = ""

            if isinstance(mllm_sub_test_output, str):
                if mllm_sub_test_output.startswith("Error:"):
                    is_error = True
                    error_reason = mllm_sub_test_output
                elif (
                    "No response generated" in mllm_sub_test_output
                    or mllm_sub_test_output == "Not Processed Yet"
                ):
                    is_error = True
                    error_reason = mllm_sub_test_output
            elif mllm_sub_test_output is None:
                is_error = True
                error_reason = "MLLM output is None"

            if is_error:
                tqdm.write(
                    f"Info: MLLM output for item {item_id}, sub-test {i} indicates an error: '{error_reason[:100]}...'. Recording as N/A for judging."
                )
                if i < len(current_item_judge_outputs):
                    current_item_judge_outputs[i] = {
                        "score_judge": "N/A",
                        "score_reason": f"Skipped judging: {error_reason}",
                    }
                continue

            # Check existing judge output for this sub-test
            judge_sub_test_output = (
                current_item_judge_outputs[i]
                if i < len(current_item_judge_outputs)
                else None
            )
            needs_judging_flag = True
            if (
                isinstance(judge_sub_test_output, dict)
                and judge_sub_test_output.get("score_judge")
                not in [None, "", "Error", "N/A"]
                and "Skipped judging"
                not in judge_sub_test_output.get("score_reason", "")
                and "Judge response decode failed"
                not in judge_sub_test_output.get("score_reason", "")
                and "API Request Failed"
                not in judge_sub_test_output.get("score_reason", "")
                and "Judge response missing key"
                not in judge_sub_test_output.get("score_reason", "")
            ):
                needs_judging_flag = False

            if (
                isinstance(judge_sub_test_output, str)
                and "Error" in judge_sub_test_output
            ):  # If previous judge result was a simple error string
                needs_judging_flag = True

            if needs_judging_flag:
                judge_ids_to_process_for_this_item.append(i)
                # tqdm.write(f"Debug: Item {item_id}, sub-test {i} needs judging. Previous judge output: {judge_sub_test_output}") # Too verbose for tqdm
            else:
                pass  # tqdm.write(f"Debug: Judge results for item {item_id}, sub-test {i} are valid. Skipping re-judging.") # Too verbose for tqdm

        if not judge_ids_to_process_for_this_item:
            tqdm.write(
                f"Info: All sub-tests for item {item_id} are either already judged satisfactorily or have MLLM outputs unsuitable for judging. Saving current state."
            )
            save_json(
                judge_results, judge_scores_output_file
            )  # Save N/A updates for this item # Use Path object
            continue

        individual_prompts_for_item = generate_individual_prompts(
            gt_item,
            mllm_responses_for_item,
            judge_ids_to_process_for_this_item,
            config_rotate_test,
        )
        all_prompts_to_judge_in_batches.extend(individual_prompts_for_item)

    total_prompts_count = len(all_prompts_to_judge_in_batches)
    if total_prompts_count == 0:
        logger.info("No new prompts to judge in this run after filtering all items.")
        # judge_results might have been updated with N/A for some items, so save it.
        save_json(judge_results, judge_scores_output_file)  # Use Path object
        return judge_results

    num_batches_to_run = (
        total_prompts_count + judge_group_size - 1
    ) // judge_group_size
    logger.info(
        f"Total prompts to judge: {total_prompts_count}, divided into {num_batches_to_run} batches of size up to {judge_group_size}."
    )

    # Add tqdm to the loop iterating through batches
    for batch_idx in tqdm(
        range(num_batches_to_run), desc="Sending batches to Judge API"
    ):
        start_index = batch_idx * judge_group_size
        end_index = min(start_index + judge_group_size, total_prompts_count)
        current_batch_prompts_info = all_prompts_to_judge_in_batches[
            start_index:end_index
        ]

        combined_user_prompt, batch_mapping = generate_prompt_batch(
            current_batch_prompts_info
        )
        # Use model_identifier from judge_model_specific_config if available

        api_payload = generate_payload_batch(
            combined_user_prompt,
            judge_model_identifier,
            judge_prompt_template,
            max_tokens_for_batch_judge,
        )

        # tqdm.write(
        #     f"Info: Sending batch {batch_idx + 1}/{num_batches_to_run} to judge API ({judge_api_url}) using model {judge_model_identifier}. Batch size: {len(current_batch_prompts_info)}"
        # )
        # logger.debug(f"Payload for batch {batch_idx + 1}: {json.dumps(api_payload, indent=2, ensure_ascii=False)}") # Can be very verbose

        try:
            api_response = requests.post(
                judge_api_url, headers=judge_headers, json=api_payload, timeout=300
            )
            api_response.raise_for_status()

            response_content_json = api_response.json()
            # logger.debug(f"Raw API response for batch {batch_idx + 1}: {json.dumps(response_content_json, indent=2, ensure_ascii=False)}")

            judge_response_text = (
                response_content_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )  # Default to empty string
            if not judge_response_text:  # Handle if content is empty or missing
                tqdm.write(
                    f"Error: No content in judge response for batch {batch_idx + 1}. Full response: {response_content_json}"
                )
                judge_response_text = (
                    '{"error": "No content from API"}'  # Make it a parsable JSON error
                )

            # Attempt to clean the response if it's wrapped in markdown code block
            if judge_response_text.strip().startswith("```"):
                # This handles ```json ... ``` and ``` ... ```
                split_text = judge_response_text.strip().split("\n", 1)
                if len(split_text) > 1:
                    judge_response_text = split_text[1].rsplit("```", 1)[0].strip()
                else:
                    # Handle case where it's just ``` or ```json without content
                    judge_response_text = (
                        judge_response_text.strip()
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )

            try:
                parsed_judge_responses = json.loads(judge_response_text)
                if not isinstance(parsed_judge_responses, dict):
                    tqdm.write(
                        f"Warning: Judge response for batch {batch_idx + 1} is not a dictionary after JSON decode. Type: {type(parsed_judge_responses)}. Raw text was: '{judge_response_text[:500]}...'"
                    )
                    raise json.JSONDecodeError(
                        "Decoded JSON is not a dictionary", judge_response_text, 0
                    )  # Raise to trigger error handling

            except json.JSONDecodeError as jde:
                tqdm.write(
                    f"Error: JSON decoding error for batch {batch_idx + 1}: {jde}. Raw text was: '{judge_response_text[:500]}...'"
                )
                for item_id_map, judge_id_map in batch_mapping:
                    if item_id_map in judge_results and judge_id_map < len(
                        judge_results[item_id_map]
                    ):
                        judge_results[item_id_map][judge_id_map] = {
                            "score_judge": "Error",
                            "score_reason": f"Judge response decode failed: {jde}",
                        }
                continue

            for i_map_idx, (item_id_map, judge_id_map) in enumerate(batch_mapping, 1):
                response_key = str(i_map_idx)
                if (
                    item_id_map not in judge_results
                    or not isinstance(judge_results[item_id_map], list)
                    or judge_id_map >= len(judge_results[item_id_map])
                ):
                    tqdm.write(
                        f"Error: Internal error: judge_results structure incorrect for item {item_id_map}, judge_id {judge_id_map} before assignment."
                    )
                    continue  # Should not happen if initialization is correct

                if response_key in parsed_judge_responses:
                    # Ensure the value for the key is a dictionary, otherwise log a warning and assign an error
                    if isinstance(parsed_judge_responses[response_key], dict):
                        judge_results[item_id_map][judge_id_map] = (
                            parsed_judge_responses[response_key]
                        )
                        log_judge_response_detail(
                            parsed_judge_responses[response_key],
                            item_id_map,
                            judge_id_map,
                        )  # Log detail
                    else:
                        tqdm.write(
                            f"Warning: Response for key '{response_key}' (item {item_id_map}, judge_id {judge_id_map}) in batch {batch_idx + 1} is not a dictionary. Value: {parsed_judge_responses[response_key]}. Marking as error."
                        )
                        judge_results[item_id_map][judge_id_map] = {
                            "score_judge": "Error",
                            "score_reason": f"Judge response value for key {response_key} is not a dictionary",
                        }
                else:
                    tqdm.write(
                        f"Warning: Missing score for key '{response_key}' (item {item_id_map}, judge_id {judge_id_map}) in batch {batch_idx + 1} response. Marking as error."
                    )
                    judge_results[item_id_map][judge_id_map] = {
                        "score_judge": "Error",
                        "score_reason": "Judge response missing key for this item in batch",
                    }

        except requests.exceptions.RequestException as req_e:
            tqdm.write(f"Error: API request failed for batch {batch_idx + 1}: {req_e}")
            for item_id_map, judge_id_map in batch_mapping:
                if item_id_map in judge_results and judge_id_map < len(
                    judge_results[item_id_map]
                ):
                    judge_results[item_id_map][judge_id_map] = {
                        "score_judge": "Error",
                        "score_reason": f"API Request Failed: {str(req_e)}",
                    }
        except Exception as e:
            tqdm.write(
                f"Error: An unexpected error occurred processing batch {batch_idx + 1}: {e}"
            )
            # tqdm.write(f"Traceback: {traceback.format_exc()}") # Optional: Add traceback for debugging unexpected errors
            for item_id_map, judge_id_map in batch_mapping:
                if item_id_map in judge_results and judge_id_map < len(
                    judge_results[item_id_map]
                ):
                    judge_results[item_id_map][judge_id_map] = {
                        "score_judge": "Error",
                        "score_reason": f"Unexpected error during batch: {str(e)}",
                    }

        finally:
            save_json(judge_results, judge_scores_output_file)
            # tqdm.write(
            #     f"Info: Judge scores saved after processing batch {batch_idx + 1} to {judge_scores_output_file}"
            # )
            if batch_idx < num_batches_to_run - 1:
                # tqdm.write(
                #     f"Info: Sleeping for {sleep_time} seconds before next batch..."
                # )
                time.sleep(sleep_time)

    logger.info(
        "Finished processing all judge batches."
    )  # Use logger directly outside the tqdm loop
    return judge_results


def combine_results_and_judgements(mllm_results_data, final_judge_scores):
    """
    Combines MLLM results with their corresponding judge scores.
    No 'config_rotate_test' needed here, as structure should align.
    """
    global logger
    combined_output = {}

    for item_id, mllm_responses_list in tqdm(
        mllm_results_data.items(), desc="Combining results"
    ):
        if not isinstance(mllm_responses_list, list):
            tqdm.write(
                f"Warning: MLLM data for item {item_id} is not a list: {mllm_responses_list}. Skipping combination for this item."
            )
            combined_output[item_id] = {
                "MLLM_answer_malformed": mllm_responses_list,
                "judge_answer": "NOT_PROCESSED_DUE_TO_MLLM_MALFORMED_DATA",
            }
            continue

        if item_id not in final_judge_scores or not isinstance(
            final_judge_scores.get(item_id), list
        ):
            tqdm.write(
                f"Warning: Judge scores not found or malformed for item_id: {item_id} during combination. MLLM responses: {len(mllm_responses_list)}"
            )
            combined_output[item_id] = [
                {
                    "MLLM_answer": mllm_ans,
                    "judge_answer": "JUDGEMENT_DATA_MISSING_OR_MALFORMED",
                }
                for mllm_ans in mllm_responses_list
            ]
            continue

        judge_scores_list = final_judge_scores[item_id]
        combined_output[item_id] = []

        num_mllm_tests = len(mllm_responses_list)
        num_judge_scores = len(judge_scores_list)

        # Iterate up to the maximum length of the two lists to capture all data,
        # though ideally they should have the same length.
        max_len = max(num_mllm_tests, num_judge_scores)
        if num_mllm_tests != num_judge_scores:
            tqdm.write(
                f"Warning: Mismatch in number of MLLM responses ({num_mllm_tests}) and judge scores ({num_judge_scores}) for item {item_id}."
            )

        for i in range(max_len):
            mllm_single_answer = (
                mllm_responses_list[i]
                if i < num_mllm_tests
                else "Error: MLLM Answer Missing for this sub-test index"
            )
            judge_single_score = (
                judge_scores_list[i]
                if i < num_judge_scores
                else {
                    "score_judge": "N/A",
                    "score_reason": "Error: Judge Score Missing for this sub-test index",
                }
            )

            combined_output[item_id].append(
                {
                    "MLLM_answer": mllm_single_answer,
                    "judge_answer": judge_single_score,
                }
            )
    return combined_output


def log_judge_response_detail(response_data_dict, item_id_str, judge_id_val):
    """Logs detailed judge response to the log file using logger, not tqdm.write."""
    global logger
    if not logger:
        return  # Should not happen if main initializes logger
    score = "N/A"
    reason = "N/A"
    if isinstance(response_data_dict, dict):
        score = response_data_dict.get("score_judge", "N/A")
        reason = response_data_dict.get("score_reason", "N/A")
    elif isinstance(response_data_dict, str):  # Handle if it's an error string
        reason = response_data_dict

    log_message = (
        f"Judge Response Detail:\n"
        f"  Item ID: {item_id_str}\n"
        f"  Sub-Test ID (judge_id): {judge_id_val}\n"
        f"  Score: {score}\n"
        f"  Reason: {reason}\n"
        f"------------------------------------"
    )
    # logger.info(log_message)  # Use logger directly to ensure it goes to the file


# =======================
# Main Processing
# =======================
def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Run ICA judging for a specified test model's results."
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
            test_model_to_judge = config["evaluation_settings"]["ica"][
                "default_model_name"
            ]

        results_base_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_base_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]

        results_base_dir.mkdir(parents=True, exist_ok=True)
        logs_base_dir.mkdir(parents=True, exist_ok=True)

        mllm_results_file = results_base_dir / f"{test_model_to_judge}_ICA_results.json"

        judge_log_file = logs_base_dir / f"{test_model_to_judge}_ICA_regular_judge.log"
        judge_scores_output_file = (
            results_base_dir / f"{test_model_to_judge}_ICA_regular_scoring.json"
        )
        # combined_output_file = (
        #     results_base_dir / f"{test_model_to_judge}_ICA_regular_combined_results.json"
        # )

        logger = setup_logger(
            f"{test_model_to_judge}_ICA_regular_judge", judge_log_file
        )
        temp_logger_for_config_errors = logger

        logger.info(f"--- Starting ICA Judging for MLLM: {test_model_to_judge} ---")
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
        os.environ["HF_DATASETS_CACHE"] = str(
            hf_cache_dir_path
        )  # Also set environment variable
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        # Load dataset from Hugging Face
        hf_dataset_name = config["general_settings"]["huggingface_dataset_name"]
        try:
            # load_dataset often has its own progress bar, so we don't wrap it with tqdm
            logger.info(
                f"Loading Hugging Face dataset '{hf_dataset_name}' from cache '{hf_config.HF_DATASETS_CACHE}'..."
            )
            dataset = load_dataset(hf_dataset_name, cache_dir=str(hf_cache_dir_path))
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

        # Get the ICA split from the dataset
        dataset_ica_split = dataset["ica"]
        logger.info(f"Using 'ica' split with {len(dataset_ica_split)} items.")

        # Convert dataset to list of dictionaries for processing
        # Add tqdm here as this iterates through the dataset split
        ground_truth_data_loaded = []
        for item in tqdm(dataset_ica_split, desc="Converting dataset to list"):
            ground_truth_data_loaded.append(dict(item))

        logger.info(
            f"Converted {len(ground_truth_data_loaded)} items from dataset to dictionary format."
        )

        mllm_results_data_loaded = load_json(mllm_results_file)
        if (
            not mllm_results_data_loaded
        ):  # Handles empty file or JSON load failure returning None/empty
            logger.error(
                f"Failed to load or empty MLLM results from {mllm_results_file}"
            )
            return
        # Ensure mllm_results_data_loaded is a dictionary mapping item_id to results list
        if not isinstance(mllm_results_data_loaded, dict):
            logger.error(
                f"MLLM results file {mllm_results_file} does not contain a valid dictionary. Found type: {type(mllm_results_data_loaded)}"
            )
            return

        final_judge_scores_data = judge_model_results_main(
            ground_truth_data_loaded,
            mllm_results_data_loaded,
            config,
            args.judge_model_name,
            judge_scores_output_file,
        )

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

        logger.info(f"--- ICA Judging finished for MLLM: {test_model_to_judge} ---")

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
        # Prefer the main logger if initialized, otherwise the temp logger, otherwise print
        log_func = (
            logger.error
            if logger
            else (
                temp_logger_for_config_errors.error
                if temp_logger_for_config_errors
                else print
            )
        )
        if logger or temp_logger_for_config_errors:
            log_func(f"An unexpected error occurred in main_judge: {e}", exc_info=True)
        else:
            log_func(f"An unexpected error occurred in main_judge: {e}")


if __name__ == "__main__":
    main()
