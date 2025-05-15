# MM-OPERA/evaluation/ICA/ICA_reasoning_judge.py

import os
import argparse
import time
import requests
import json
from pathlib import Path
from tqdm import tqdm

# Relative imports
from ..config_loader import get_config, get_api_key, PROJECT_ROOT
from ..utils import setup_logger, save_json, load_json

# Hugging Face datasets library
from datasets import load_dataset, config as hf_config

# Global logger
logger = None


def get_image_descriptions_for_permutation(dataset_item, permutation_index: int):
    """
    Gets the correct three input image descriptions for the MLLM based on the permutation.
    Permutations:
    0: (img1, img2, img3) -> MLLM proposes img4_analog_to_img2
    1: (img2, img1, img4) -> MLLM proposes img3_analog_to_img1
    2: (img3, img4, img1) -> MLLM proposes img2_analog_to_img4
    3: (img4, img3, img2) -> MLLM proposes img1_analog_to_img3
    """
    # Ensure all descriptions are available
    desc = [
        dataset_item.get("description1"),
        dataset_item.get("description2"),
        dataset_item.get("description3"),
        dataset_item.get("description4"),
    ]
    if any(d is None for d in desc):
        # This might happen if a dataset item doesn't have 4 descriptions.
        # The ICA dataset example provided has description1-4.
        logger.warning(
            f"Dataset item {dataset_item.get('id')} missing one or more image descriptions (1-4)."
        )
        return None, None, None, None  # Indicate error or incomplete data

    if permutation_index == 0:  # Standard: 1,2,3 -> expect 4
        return desc[0], desc[1], desc[2], desc[3]
    elif permutation_index == 1:  # Rotated: 2,1,4 -> expect 3
        return desc[1], desc[0], desc[3], desc[2]
    elif permutation_index == 2:  # Rotated: 3,4,1 -> expect 2
        return desc[2], desc[3], desc[0], desc[1]
    elif permutation_index == 3:  # Rotated: 4,3,2 -> expect 1
        return desc[3], desc[2], desc[1], desc[0]
    else:
        raise ValueError(f"Invalid permutation_index: {permutation_index}")


def format_judge_prompt_item(
    dataset_item, mllm_output_text: str, permutation_index: int
) -> str | None:
    """
    Formats a single item for the ICA judge's prompt, considering the permutation.
    dataset_item: An item from the Hugging Face 'ica' split.
    mllm_output_text: The raw text output from the MLLM for this specific permutation.
    permutation_index: 0, 1, 2, or 3, indicating which MLLM output (from rotated tests) is being judged.
    """

    # Get the descriptions for the three images *input* to MLLM for this permutation,
    # and the *ground truth* Image 4 for this permutation.
    (
        mllm_input_img1_desc,
        mllm_input_img2_desc,
        mllm_input_img3_desc,
        gt_img4_for_this_perm_desc,
    ) = get_image_descriptions_for_permutation(dataset_item, permutation_index)

    if mllm_input_img1_desc is None:  # Error getting descriptions
        return None

    # The 'reasoning' field in ICA dataset is a JSON string representing a list of dicts
    # Each dict has 'pair_id', 'explanation', 'path'
    try:
        # ICA 'reasoning' is a list of two dictionaries, one for each pair.
        # We need to pick the correct pair explanations/paths for the current permutation.
        reasoning_list_str = dataset_item.get("reasoning")
        if not reasoning_list_str:
            logger.warning(
                f"Missing 'reasoning' field in dataset item: {dataset_item.get('id')}"
            )
            return None  # Cannot proceed without reference reasoning paths

        # Safely parse the reasoning string
        # The example shows it's a JSON string, so json.loads is appropriate
        reasoning_data = json.loads(reasoning_list_str)
        if not isinstance(reasoning_data, list) or len(reasoning_data) < 2:
            logger.error(
                f"Invalid 'reasoning' format for {dataset_item.get('id')}. Expected list of 2 dicts. Got: {reasoning_data}"
            )
            return None

        # For ICA, the reference paths are always for Pair 1 (orig_img1, orig_img2) and Pair 2 (orig_img3, orig_img4)
        # The judge prompt expects reference for the *current* MLLM input pairs.
        # This requires careful mapping if the original dataset's reasoning is fixed to the 0th permutation.
        # For simplicity, let's assume dataset_item["reasoning"] gives paths for the 0-th permutation (img1-img2, img3-img4)
        # And we adapt them based on `permutation_index`.
        # This part is tricky and depends heavily on how `ICA_run.py` structures its requests and how `dataset_item["reasoning"]` is structured.
        # The original script's `generate_individual_prompts` logic for ICA was:
        # explanation1 = exp1 if judge_id < 2 else exp2
        # explanation2 = exp2 if judge_id < 2 else exp1
        # This suggests `reasoning_data[0]` is for "first pair" and `reasoning_data[1]` is for "second pair" OF THE ORIGINAL 0-th permutation.

        ref_expl_orig_pair1 = reasoning_data[0].get("explanation", "N/A")
        ref_path_orig_pair1 = reasoning_data[0].get("path", "N/A")
        ref_expl_orig_pair2 = reasoning_data[1].get("explanation", "N/A")
        ref_path_orig_pair2 = reasoning_data[1].get("path", "N/A")

        # Determine which original pair corresponds to the MLLM's current Pair 1 and Pair 2
        if (
            permutation_index == 0 or permutation_index == 1
        ):  # MLLM's Pair1 is an original pair (1-2 or 2-1), MLLM's Pair2 is an original pair (3-4 or 4-3)
            ref_expl_mllm_pair1 = ref_expl_orig_pair1
            ref_path_mllm_pair1 = ref_path_orig_pair1
            ref_expl_mllm_pair2 = ref_expl_orig_pair2
            ref_path_mllm_pair2 = ref_path_orig_pair2
        elif (
            permutation_index == 2 or permutation_index == 3
        ):  # MLLM's Pair1 is an original pair (3-4 or 4-3), MLLM's Pair2 is an original pair (1-2 or 2-1)
            ref_expl_mllm_pair1 = ref_expl_orig_pair2
            ref_path_mllm_pair1 = ref_path_orig_pair2
            ref_expl_mllm_pair2 = ref_expl_orig_pair1
            ref_path_mllm_pair2 = ref_path_orig_pair1
        else:  # Should not happen
            ref_expl_mllm_pair1, ref_path_mllm_pair1 = "Error determining ref", "Error"
            ref_expl_mllm_pair2, ref_path_mllm_pair2 = "Error determining ref", "Error"

    except (json.JSONDecodeError, TypeError, IndexError) as e:
        logger.error(
            f"Error parsing 'reasoning' field for {dataset_item.get('id')}: {e}. Content: {dataset_item.get('reasoning')}"
        )
        return None

    ref_relation = dataset_item.get("relation", "N/A")  # Overall relation

    user_prompt = (
        f"Problem:\n"
        f"- Image 1: {mllm_input_img1_desc}\n"
        f"- Image 2: {mllm_input_img2_desc}\n"
        f"- Image 3: {mllm_input_img3_desc}\n"
        f"Reference Answer:\n"
        f"- Image 4: {gt_img4_for_this_perm_desc}\n"
        f"- Relation: {ref_relation}\n"
        f"- Explanation 1: {ref_expl_mllm_pair1}\n"
        f"- Association Path 1: {ref_path_mllm_pair1}\n"
        f"- Explanation 2: {ref_expl_mllm_pair2}\n"
        f"- Association Path 2: {ref_path_mllm_pair2}\n"
        f"MLLM's Output:\n{mllm_output_text} \n"
    )
    return user_prompt


def validate_ica_judgement(judgement_data: dict) -> bool:
    """Validates structure of a single ICA judgement."""
    if not isinstance(judgement_data, dict):
        return False
    keys_needed = [
        "path1",
        "path2",
        "hop_quality_path1",
        "hop_quality_path2",
        "explanation",
    ]
    if not all(k in judgement_data for k in keys_needed):
        return False
    if not isinstance(judgement_data["path1"], str):
        return False
    if not isinstance(judgement_data["path2"], str):
        return False
    if not isinstance(judgement_data["explanation"], str):
        return False

    for hop_quality_key in ["hop_quality_path1", "hop_quality_path2"]:
        if not isinstance(judgement_data[hop_quality_key], dict):
            return False
        for hop, scores in judgement_data[hop_quality_key].items():
            if not isinstance(hop, str):
                return False
            if not isinstance(scores, list) or len(scores) != 3:
                return False
            if not (isinstance(scores[0], (int, float)) and 0 <= scores[0] <= 1):
                return False
            if not (isinstance(scores[1], (int, float)) and 0 <= scores[1] <= 1):
                return False
            if not (isinstance(scores[2], int) and scores[2] in [0, 1]):
                return False
    return True


def perform_ica_reasoning_judgment(
    model_to_judge_name: str,
    judge_model_name: str,
    dataset_ica_split,  # This is dataset["ica"]
    mllm_results: dict,  # {id: [output_perm0, output_perm1, ...]}
    config: dict,
    output_file_path: Path,
):
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
    ica_judge_settings = config["evaluation_settings"]["ica"]["reasoning_judge"]
    system_prompt = ica_judge_settings["prompt"]
    group_size = ica_judge_settings["judge_group_size"]
    judge_max_tokens = (
        ica_judge_settings["judge_max_tokens_per_batch_multiplier"] * group_size
    )
    sleep_time = ica_judge_settings.get(
        "sleep_time_after_judge_api",
        general_config.get("sleep_time_between_judge_requests", 7),
    )
    expect_rotated = ica_judge_settings.get("expect_rotated_test_results", True)
    num_permutations_to_judge = 4 if expect_rotated else 1

    existing_judgements = load_json(output_file_path) or {}

    logger.info(
        f"Loaded {sum(len(v) for v in existing_judgements.values() if isinstance(v, list))} existing judgement entries."
    )

    items_for_judge_api = (
        []
    )  # List of dicts: {id, perm_idx, dataset_item_ref, mllm_output_text}

    for hf_item in dataset_ica_split:
        item_id = hf_item.get("id")
        if not item_id:
            logger.warning(f"Skipping dataset item due to missing 'id': {hf_item}")
            continue

        mllm_outputs_for_item = mllm_results.get(item_id)
        if not mllm_outputs_for_item or not isinstance(mllm_outputs_for_item, list):
            # logger.warning(f"No MLLM result list found for id '{item_id}'. Skipping.")
            continue

        if len(mllm_outputs_for_item) < num_permutations_to_judge:
            logger.warning(
                f"MLLM results for id '{item_id}' has {len(mllm_outputs_for_item)} items, expected {num_permutations_to_judge}. Some permutations might be skipped or marked as error."
            )

        if item_id not in existing_judgements:
            existing_judgements[item_id] = [None] * num_permutations_to_judge
        elif len(existing_judgements[item_id]) < num_permutations_to_judge:
            existing_judgements[item_id].extend(
                [None] * (num_permutations_to_judge - len(existing_judgements[item_id]))
            )

        for perm_idx in range(num_permutations_to_judge):
            # 修改这里：检查是否为None或包含error字段
            if existing_judgements[item_id][perm_idx] is not None and not (
                isinstance(existing_judgements[item_id][perm_idx], dict)
                and "error" in existing_judgements[item_id][perm_idx]
            ):
                continue

            if perm_idx >= len(mllm_outputs_for_item):
                logger.warning(
                    f"Missing MLLM output for id '{item_id}', permutation {perm_idx}. Marking as error."
                )
                existing_judgements[item_id][perm_idx] = {
                    "error": f"Missing MLLM output for permutation {perm_idx}."
                }
                continue

            mllm_output_text = mllm_outputs_for_item[perm_idx]
            if (
                not isinstance(mllm_output_text, str)
                or "Error:" in mllm_output_text
                or not mllm_output_text.strip()
            ):
                logger.warning(
                    f"Invalid/error MLLM output for id '{item_id}', perm {perm_idx}. Output: {str(mllm_output_text)[:100]}"
                )
                existing_judgements[item_id][perm_idx] = {
                    "error": "Invalid or error MLLM output for this permutation.",
                    "mllm_output_preview": str(mllm_output_text)[:100],
                }
                continue

            items_for_judge_api.append(
                {
                    "id": item_id,
                    "perm_idx": perm_idx,
                    "dataset_item_ref": hf_item,  # Keep a reference to the full HF dataset item
                    "mllm_output_text": mllm_output_text,
                }
            )

    if not items_for_judge_api:
        logger.info("No new items or permutations to judge for ICA.")
        save_json(existing_judgements, output_file_path)
        return

    logger.info(
        f"Found {len(items_for_judge_api)} new (item, permutation) pairs for ICA reasoning judgement."
    )

    with tqdm(
        total=len(items_for_judge_api),
        desc=f"Judging ICA for {model_to_judge_name} with {judge_model_name}",
    ) as pbar:
        for i in range(0, len(items_for_judge_api), group_size):
            batch_items_data = items_for_judge_api[i : i + group_size]

            user_prompts_for_batch_api = []
            current_batch_identifiers = []  # List of (id, perm_idx)

            for idx_in_batch, item_data in enumerate(batch_items_data):
                formatted_prompt_str = format_judge_prompt_item(
                    item_data["dataset_item_ref"],
                    item_data["mllm_output_text"],
                    item_data["perm_idx"],
                )
                if formatted_prompt_str is None:
                    logger.error(
                        f"Failed to format prompt for id {item_data['id']}, perm {item_data['perm_idx']}. Skipping this item in batch."
                    )
                    existing_judgements[item_data["id"]][item_data["perm_idx"]] = {
                        "error": "Failed to format judge prompt."
                    }
                    continue

                user_prompts_for_batch_api.append(
                    f"{idx_in_batch + 1}.\n{formatted_prompt_str}"
                )
                current_batch_identifiers.append(
                    (item_data["id"], item_data["perm_idx"])
                )

            if not user_prompts_for_batch_api:
                pbar.update(len(batch_items_data))
                save_json(existing_judgements, output_file_path)
                continue

            full_user_prompt = "\n\n".join(user_prompts_for_batch_api)
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
                    f"Sending payload for batch (first item: {current_batch_identifiers[0] if current_batch_identifiers else 'N/A'}) to {judge_api_url}"
                )
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
                    raise ValueError("Judge model returned no content.")

                if judge_response_content.startswith("```json"):
                    judge_response_content = judge_response_content[7:]
                if judge_response_content.endswith("```"):
                    judge_response_content = judge_response_content[:-3]
                parsed_judgements_batch = json.loads(judge_response_content.strip())

                for idx_in_api_batch, (item_id, perm_idx) in enumerate(
                    current_batch_identifiers
                ):
                    judgement_key = str(idx_in_api_batch + 1)
                    if judgement_key in parsed_judgements_batch:
                        single_judgement = parsed_judgements_batch[judgement_key]
                        if validate_ica_judgement(single_judgement):
                            existing_judgements[item_id][perm_idx] = single_judgement
                            # logger.info(
                            #     f"Successfully judged: id {item_id}, perm_idx {perm_idx}"
                            # )
                        else:
                            logger.error(
                                f"Invalid judgement structure for id {item_id}, perm {perm_idx}: {single_judgement}"
                            )
                            existing_judgements[item_id][perm_idx] = {
                                "error": "Invalid judgement structure from judge.",
                                "raw_judgement": single_judgement,
                            }
                    else:
                        logger.error(
                            f"Judge response missing key '{judgement_key}' for id {item_id}, perm {perm_idx}."
                        )
                        existing_judgements[item_id][perm_idx] = {
                            "error": f"Judge response missing key '{judgement_key}'."
                        }

            except Exception as e:
                logger.error(
                    f"Error processing API batch (first item: {current_batch_identifiers[0] if current_batch_identifiers else 'N/A'}): {e}",
                    exc_info=True,
                )
                raw_resp_text = (
                    api_response.text
                    if "api_response" in locals() and hasattr(api_response, "text")
                    else "No response text available"
                )
                for (
                    item_id,
                    perm_idx,
                ) in current_batch_identifiers:
                    if existing_judgements[item_id][perm_idx] is None:
                        existing_judgements[item_id][perm_idx] = {
                            "error": f"API/Processing error in batch: {str(e)[:100]}",
                            "raw_response_preview": raw_resp_text[:200],
                        }
            finally:
                save_json(existing_judgements, output_file_path)
                pbar.update(len(batch_items_data))
                if i + group_size < len(items_for_judge_api):
                    time.sleep(sleep_time)

    logger.info(
        f"--- ICA Reasoning judgement finished for model: {model_to_judge_name} ---"
    )


def main():
    global logger
    parser = argparse.ArgumentParser(description="Perform ICA reasoning judgement.")
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
        config = get_config()
        model_to_judge_name = args.test_model_name
        if not model_to_judge_name:
            model_to_judge_name = config["evaluation_settings"]["ica"][
                "default_model_name"
            ]
        judge_model_name_arg = args.judge_model_name
        if not judge_model_name_arg:
            judge_model_name_arg = config["evaluation_settings"]["ica"][
                "reasoning_judge"
            ].get("default_judge_model_name")
            if not judge_model_name_arg:
                raise ValueError(
                    "Judge model name not provided via argument and not found in config's default_judge_model_name."
                )

        # Paths
        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"{model_to_judge_name}_ICA_reasoning_judge.log"
        mllm_results_file = (
            results_dir / f"{model_to_judge_name}_ICA_results.json"
        )  # Input MLLM answers
        output_scoring_file = (
            results_dir / f"{model_to_judge_name}_ICA_reasoning_scoring.json"
        )  # Output scores

        logger = setup_logger(f"{model_to_judge_name}_ICA_ReasoningJudge", log_file)
        logger.info(
            f"--- Starting ICA reasoning judgement for MLLM: {model_to_judge_name} using Judge: {judge_model_name_arg} ---"
        )
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"MLLM results (input): {mllm_results_file}")
        logger.info(f"Reasoning scores (output): {output_scoring_file}")

        if not mllm_results_file.exists():
            logger.error(
                f"MLLM results file not found: {mllm_results_file}. Run ICA_run.py first."
            )
            return
        mllm_results = load_json(mllm_results_file)
        if not mllm_results or not isinstance(mllm_results, dict):
            logger.error(f"Failed to load/parse MLLM results from {mllm_results_file}.")
            return

        # Dataset
        hf_cache = str(
            PROJECT_ROOT / config["general_settings"]["huggingface_cache_dir"]
        )
        hf_ds_name = config["general_settings"]["huggingface_dataset_name"]
        hf_config.HF_DATASETS_CACHE = hf_cache
        os.environ["HF_DATASETS_CACHE"] = hf_cache
        Path(hf_cache).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Hugging Face cache directory set to: {hf_config.HF_DATASETS_CACHE}"
        )

        ica_columns_needed = [
            "id",
            "foldername",
            "description1",
            "description2",
            "description3",
            "description4",
            "relation",
            "reasoning",
        ]

        dataset = load_dataset(hf_ds_name, cache_dir=hf_cache)
        if "ica" not in dataset:
            logger.error("'ica' split not found.")
            return

        # Attempt to format dataset to load only necessary columns
        try:
            available_cols = set(dataset["ica"].column_names)
            if not all(col in available_cols for col in ica_columns_needed):
                missing = [
                    col for col in ica_columns_needed if col not in available_cols
                ]
                logger.error(
                    f"Required columns missing from ICA dataset: {missing}. Cannot proceed with selective column loading for judge."
                )

                dataset_ica_split_formatted = dataset["ica"]
                logger.warning(
                    "Proceeding with all columns for ICA dataset. Image decoding errors may occur."
                )
            else:
                dataset_ica_split_formatted = dataset["ica"].with_format(
                    "python", columns=ica_columns_needed
                )
                logger.info(
                    f"ICA dataset formatted to load only specified columns: {ica_columns_needed}"
                )
        except Exception as e:
            logger.error(
                f"Failed to format ICA dataset with specific columns: {e}. Using full dataset, image errors may occur."
            )
            dataset_ica_split_formatted = dataset["ica"]

        perform_ica_reasoning_judgment(
            model_to_judge_name,
            judge_model_name_arg,
            dataset_ica_split_formatted,
            mllm_results,
            config,
            output_scoring_file,
        )

    except Exception as e:
        if logger:
            logger.critical(
                f"Critical error in ICA_reasoning_judge main: {e}", exc_info=True
            )
        else:
            print(f"Critical error: {e}")


if __name__ == "__main__":
    main()
