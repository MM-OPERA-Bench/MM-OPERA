# MM-OPERA/evaluation/RIA/RIA_reasoning_judge.py

import os
import argparse
import time
import requests
import json
from pathlib import Path
from tqdm import tqdm

# Relative imports for modules within the 'evaluation' package
from ..config_loader import get_config, get_api_key, PROJECT_ROOT
from ..utils import setup_logger, save_json, load_json

# Hugging Face datasets library
from datasets import load_dataset, config as hf_config

# Global logger, will be initialized in main
logger = None


def format_judge_prompt_item(dataset_item, mllm_output_text: str) -> str:
    """
    Formats a single item for the judge's prompt.
    """
    # Extract concepts from dataset item descriptions (or MLLM output if preferred)
    # For consistency with the example, let's use dataset descriptions.
    # The HuggingFace dataset has 'description1', 'description2', etc.
    concept1 = dataset_item.get("description1", "Unknown Concept 1")
    concept2 = dataset_item.get("description2", "Unknown Concept 2")

    # Reference answer components from the dataset item
    ref_relation = dataset_item.get("relation", "N/A")
    ref_explanation = dataset_item.get("explanation", "N/A")
    ref_path = dataset_item.get(
        "reasoning", "N/A"
    )  # 'reasoning' field holds the path in HF dataset

    user_prompt = f"Problem:\n- First image: {concept1}\n- Second image: {concept2}\nReference Answer:\n- Relation: {ref_relation}\n- Explanation: {ref_explanation}\n- Association Path: {ref_path}\nMLLM Output:\n{mllm_output_text}"
    return user_prompt


def validate_judgement(judgement_data: dict) -> bool:
    """
    Validates the structure of a single judgement from the LLM judge.
    """
    if not isinstance(judgement_data, dict):
        return False
    if not all(k in judgement_data for k in ["path", "hop_quality", "explanation"]):
        return False
    if not isinstance(judgement_data["path"], str):
        return False
    if not isinstance(judgement_data["explanation"], str):
        return False
    if not isinstance(judgement_data["hop_quality"], dict):  # hop_quality is a dict
        return False

    for hop, scores in judgement_data["hop_quality"].items():
        if not isinstance(hop, str):
            return False
        if not isinstance(scores, list) or len(scores) != 3:
            return False
        if not (isinstance(scores[0], (int, float)) and 0 <= scores[0] <= 1):
            return False  # Reasonable
        if not (isinstance(scores[1], (int, float)) and 0 <= scores[1] <= 1):
            return False  # Precise
        if not (isinstance(scores[2], int) and scores[2] in [0, 1]):
            return False  # Knowledgeable
    return True


def perform_reasoning_judgment(
    model_to_judge_name: str,
    judge_model_name: str,
    dataset_ria_split,
    mllm_results: dict,
    config: dict,
    output_file_path: Path,
):
    """
    Main loop to get reasoning judgements from the judge model.
    """
    global logger
    if logger is None:
        print("Error: Logger not initialized.")
        return

    judge_model_config = config["models"].get(judge_model_name)
    if not judge_model_config:
        logger.error(f"Configuration for judge model '{judge_model_name}' not found.")
        return

    provider_name = judge_model_config["provider"]
    provider_config = config["api_providers"].get(provider_name)
    if not provider_config:
        logger.error(
            f"Configuration for provider '{provider_name}' (for judge model) not found."
        )
        return

    judge_api_key = get_api_key(judge_model_name)
    if not judge_api_key:
        logger.error(
            f"API key for judge model '{judge_model_name}' not found. Skipping."
        )
        return

    judge_base_url = provider_config["base_url"]
    judge_endpoint = judge_model_config.get("endpoint", "/chat/completions")
    judge_api_url = f"{judge_base_url.rstrip('/')}/{judge_endpoint.lstrip('/')}"

    headers = {
        "Authorization": f"Bearer {judge_api_key}",
        "Content-Type": "application/json",
    }

    judge_model_identifier = (
        judge_model_config.get("model_identifier") or judge_model_name
    )

    general_config = config["general_settings"]
    ria_judge_settings = config["evaluation_settings"]["ria"]["reasoning_judge"]
    system_prompt = ria_judge_settings["prompt"]
    group_size = ria_judge_settings["judge_group_size"]
    # Calculate max_tokens for the batch based on group_size
    # The multiplier is per item, so total max_tokens is multiplier * group_size
    judge_max_tokens = (
        ria_judge_settings["judge_max_tokens_per_batch_multiplier"] * group_size
    )
    sleep_time = ria_judge_settings.get(
        "sleep_time_after_judge_api",
        general_config.get("sleep_time_between_judge_requests", 7),
    )

    # Load existing results
    existing_judgements = load_json(output_file_path) or {}
    processed_item_ids = set(existing_judgements.keys())

    logger.info(f"Loaded {len(existing_judgements)} existing judgements.")

    items_to_judge = []
    for hf_item in dataset_ria_split:
        item_id = hf_item.get("id")
        if not item_id:
            logger.warning(f"Skipping dataset item due to missing 'id': {hf_item}")
            continue

        if item_id in processed_item_ids:
            continue

        if item_id not in mllm_results:
            continue

        mllm_output_text = mllm_results[item_id]
        if (
            not isinstance(mllm_output_text, str)
            or "Error:" in mllm_output_text
            or not mllm_output_text.strip()
        ):
            logger.warning(
                f"Skipping judgement for '{item_id}' due to invalid or error MLLM output: {mllm_output_text[:100]}..."
            )
            existing_judgements[item_id] = {
                "error": "Invalid or error MLLM output",
                "mllm_output_preview": mllm_output_text[:100],
            }
            continue

        items_to_judge.append(
            {
                "id": item_id,
                "dataset_item": hf_item,
                "mllm_output_text": mllm_output_text,
            }
        )

    if not items_to_judge:
        logger.info("No new items to judge.")
        save_json(existing_judgements, output_file_path)
        return

    logger.info(
        f"Found {len(items_to_judge)} new items to send for reasoning judgement."
    )

    with tqdm(
        total=len(items_to_judge),
        desc=f"Judging {model_to_judge_name} with {judge_model_name}",
    ) as pbar:
        for i in range(0, len(items_to_judge), group_size):
            batch = items_to_judge[i : i + group_size]

            all_in_batch_processed = True
            for item_data in batch:
                if item_data["id"] not in processed_item_ids:
                    all_in_batch_processed = False
                    break
            if all_in_batch_processed:
                # logger.info(
                #     f"Skipping batch starting with '{batch[0]['id']}' as all items seem processed."
                # )
                pbar.update(len(batch))
                continue

            user_prompts_for_batch = []
            current_batch_ids = []
            for idx_in_batch, item_data in enumerate(batch):
                user_prompt_item_str = format_judge_prompt_item(
                    item_data["dataset_item"], item_data["mllm_output_text"]
                )
                user_prompts_for_batch.append(
                    f"{idx_in_batch + 1}.\n{user_prompt_item_str}"
                )
                current_batch_ids.append(item_data["id"])

            full_user_prompt = "\n\n".join(user_prompts_for_batch)

            payload = {
                "model": judge_model_identifier,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt},
                ],
                "max_tokens": judge_max_tokens,
            }

            try:
                logger.debug(
                    f"Sending payload for batch: {current_batch_ids} to {judge_api_url}"
                )
                # logger.debug(f"Payload content (first 200 chars of user prompt): {full_user_prompt[:200]}")

                api_response = requests.post(
                    judge_api_url, headers=headers, json=payload, timeout=240
                )
                api_response.raise_for_status()
                response_json = api_response.json()
                judge_response_content = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )

                if not judge_response_content:
                    logger.error(
                        f"No content in judge model response for batch {current_batch_ids}. Response: {response_json}"
                    )
                    for item_id in current_batch_ids:
                        existing_judgements[item_id] = {
                            "error": "Judge model returned no content."
                        }
                else:
                    # The judge model is expected to return a JSON string
                    # Remove potential markdown code fences
                    if judge_response_content.startswith("```json"):
                        judge_response_content = judge_response_content[7:]
                    if judge_response_content.endswith("```"):
                        judge_response_content = judge_response_content[:-3]

                    try:
                        parsed_judgements = json.loads(judge_response_content.strip())
                        if not isinstance(parsed_judgements, dict):
                            raise json.JSONDecodeError(
                                "Judge response is not a dictionary",
                                judge_response_content,
                                0,
                            )

                        for idx_in_batch, original_item_id in enumerate(
                            current_batch_ids
                        ):
                            judgement_key = str(
                                idx_in_batch + 1
                            )  # Judge returns keys "1", "2", ...
                            if judgement_key in parsed_judgements:
                                single_judgement = parsed_judgements[judgement_key]
                                if validate_judgement(single_judgement):
                                    existing_judgements[original_item_id] = (
                                        single_judgement
                                    )
                                    # logger.info(
                                    #     f"Successfully judged and validated: {original_item_id}"
                                    # )
                                else:
                                    logger.error(
                                        f"Invalid judgement structure for {original_item_id}: {single_judgement}"
                                    )
                                    # existing_judgements[original_item_id] = {
                                    #     "error": "Invalid judgement structure from judge model.",
                                    #     "raw_judgement": single_judgement,
                                    # }
                            else:
                                logger.error(
                                    f"Judge response missing key '{judgement_key}' for item {original_item_id} in batch."
                                )
                                # existing_judgements[original_item_id] = {
                                #     "error": f"Judge response missing key '{judgement_key}'."
                                # }

                    except json.JSONDecodeError as json_e:
                        logger.error(
                            f"Failed to parse JSON from judge model for batch {current_batch_ids}: {json_e}"
                        )
                        logger.error(f"Raw judge response: {judge_response_content}")
                        # for item_id in current_batch_ids:
                        #     existing_judgements[item_id] = {
                        #         "error": "Failed to parse JSON from judge model.",
                        #         "raw_response": judge_response_content,
                        #     }

            except requests.exceptions.HTTPError as http_err:
                logger.error(
                    f"HTTP error occurred for batch {current_batch_ids}: {http_err} - Response: {api_response.text}"
                )
                # for item_id in current_batch_ids:
                #     existing_judgements[item_id] = {
                #         "error": f"HTTP error from judge API: {http_err}",
                #         "details": api_response.text[:200],
                #     }
            except requests.exceptions.RequestException as req_err:
                logger.error(
                    f"Request exception for batch {current_batch_ids}: {req_err}"
                )
                # for item_id in current_batch_ids:
                #     existing_judgements[item_id] = {
                #         "error": f"Request exception: {req_err}"
                #     }
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred processing batch {current_batch_ids}: {e}",
                    exc_info=True,
                )
                # for item_id in current_batch_ids:
                #     existing_judgements[item_id] = {
                #         "error": f"Unexpected error: {str(e)}"
                #     }
            finally:
                # 每批次后保存结果
                save_json(existing_judgements, output_file_path)
                # 更新已处理项目集合
                processed_item_ids = set(existing_judgements.keys())
                pbar.update(len(batch))
                if i + group_size < len(
                    items_to_judge
                ):  # Don't sleep after the last batch
                    # logger.info(f"Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)

    logger.info(f"Final reasoning scores saved to {output_file_path}")
    logger.info(
        f"--- Reasoning judgement finished for model: {model_to_judge_name} ---"
    )


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Perform reasoning judgement on RIA MLLM outputs."
    )
    parser.add_argument(
        "--test_model_name",
        type=str,
        # required=True,
        help="Name of the MLLM whose results are being judged (e.g., Gemini-2.0-Flash-Thinking-Exp). Must match a model defined in model_config.yaml.",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default=None,
        help="Name of the LLM to use as the judge (e.g., GPT-4o-judge). Must be defined in model_config.yaml.",
    )
    args = parser.parse_args()

    try:
        # --- 1. Load Configuration ---
        config = get_config()
        test_model_to_judge = args.test_model_name
        if not test_model_to_judge:
            test_model_to_judge = config["evaluation_settings"]["ria"][
                "default_model_name"
            ]
        judge_model_name_arg = args.judge_model_name
        if not judge_model_name_arg:
            judge_model_name_arg = config["evaluation_settings"]["ria"][
                "reasoning_judge"
            ].get("default_judge_model_name")
            if not judge_model_name_arg:
                raise ValueError(
                    "Judge model name not provided via argument and not found in config's default_judge_model_name."
                )
        final_judge_model_name = judge_model_name_arg

        # --- 2. Setup Paths ---
        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]

        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = logs_dir / f"{test_model_to_judge}_RIA_reasoning_judge.log"
        # Input file: results from RIA_run.py for the model being judged
        mllm_results_input_path = (
            results_dir / f"{test_model_to_judge}_RIA_results.json"
        )
        # Output file for reasoning scores
        reasoning_output_path = (
            results_dir / f"{test_model_to_judge}_RIA_reasoning_scoring.json"
        )

        # --- 3. Setup Logger ---
        logger = setup_logger(
            f"{test_model_to_judge}_RIA_reasoning_judge", log_file_path
        )
        logger.info(
            f"--- Starting RIA reasoning judgement for MLLM: {test_model_to_judge} using Judge: {final_judge_model_name} ---"
        )
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"MLLM results (input): {mllm_results_input_path}")
        logger.info(f"Reasoning scores (output): {reasoning_output_path}")

        # --- 4. Load MLLM Results ---
        if not mllm_results_input_path.exists():
            logger.error(
                f"MLLM results file not found: {mllm_results_input_path}. Please run RIA_run.py for '{test_model_to_judge}' first."
            )
            return
        mllm_results = load_json(mllm_results_input_path)
        if not mllm_results or not isinstance(mllm_results, dict):
            logger.error(
                f"Failed to load or parse MLLM results from {mllm_results_input_path}, or it's empty/not a dict."
            )
            return
        logger.info(
            f"Loaded {len(mllm_results)} MLLM results from {mllm_results_input_path}."
        )

        # --- 5. Load Hugging Face Dataset ---
        hf_cache_dir_str = str(
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_dataset_name = config["general_settings"]["huggingface_dataset_name"]

        hf_config.HF_DATASETS_CACHE = hf_cache_dir_str
        os.environ["HF_DATASETS_CACHE"] = hf_cache_dir_str
        Path(hf_cache_dir_str).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        try:
            dataset = load_dataset(
                hf_dataset_name, cache_dir=hf_cache_dir_str
            )  # Explicitly pass cache_dir
            logger.info(f"Dataset '{hf_dataset_name}' loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face dataset '{hf_dataset_name}': {e}",
                exc_info=True,
            )
            return

        if "ria" not in dataset:
            logger.error(
                f"'ria' split not found in dataset '{hf_dataset_name}'. Available: {list(dataset.keys())}"
            )
            return

        columns_needed_by_judge_script = [
            "id",
            "description1",
            "description2",
            "relation",
            "explanation",
            "reasoning",  # This is the 'path' in the reference answer
        ]
        try:
            # Check if all needed columns exist in the dataset features
            available_columns = set(dataset["ria"].column_names)
            missing_columns = [
                col
                for col in columns_needed_by_judge_script
                if col not in available_columns
            ]
            if missing_columns:
                logger.error(
                    f"The following required columns are missing from the dataset's 'ria' split: {missing_columns}. Available: {list(available_columns)}"
                )
                return

            dataset_ria_split = dataset["ria"].with_format(
                "python", columns=columns_needed_by_judge_script
            )
            logger.info(
                f"Using 'ria' split, formatted to load only columns: {columns_needed_by_judge_script}"
            )
            logger.info(
                f"Number of items in 'ria' split after formatting: {len(dataset_ria_split)}"
            )

        except Exception as e:
            logger.error(
                f"Error when trying to format dataset or select columns: {e}",
                exc_info=True,
            )
            # Fallback to loading all columns if formatting fails, but warn loudly
            logger.warning(
                "Falling back to loading all columns from 'ria' split. Image decoding errors might occur."
            )
            dataset_ria_split = dataset["ria"]

        # --- 6. Perform Judgement ---
        perform_reasoning_judgment(
            model_to_judge_name=test_model_to_judge,
            judge_model_name=final_judge_model_name,
            dataset_ria_split=dataset_ria_split,
            mllm_results=mllm_results,
            config=config,
            output_file_path=reasoning_output_path,
        )

    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found error in main: {e}", exc_info=True)
        else:
            print(f"File not found error: {e}")
    except ValueError as e:
        if logger:
            logger.error(f"Value error in main: {e}", exc_info=True)
        else:
            print(f"Value error: {e}")
    except Exception as e:
        if logger:
            logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
        else:
            print(f"An unexpected error occurred in main: {e}")


if __name__ == "__main__":
    # To run (from MM-OPERA project root):
    # python -m evaluation.RIA.RIA_reasoning_judge --model_name gpt-4o
    # (This will judge gpt-4o's RIA outputs using the default judge model from config)
    #
    # Or specify the judge model:
    # python -m evaluation.RIA.RIA_reasoning_judge --model_name qwen-vl-plus-0809 --judge_model_name gpt-4o-2024-08-06
    main()
