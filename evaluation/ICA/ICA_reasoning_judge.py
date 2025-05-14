# MM-OPERA/evaluation/ICA/ICA_reasoning_judge.py

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
            f"Dataset item {dataset_item.get('foldername')} missing one or more image descriptions (1-4)."
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
                f"Missing 'reasoning' field in dataset item: {dataset_item.get('foldername')}"
            )
            return None  # Cannot proceed without reference reasoning paths

        # Safely parse the reasoning string
        # The example shows it's a JSON string, so json.loads is appropriate
        reasoning_data = json.loads(reasoning_list_str)
        if not isinstance(reasoning_data, list) or len(reasoning_data) < 2:
            logger.error(
                f"Invalid 'reasoning' format for {dataset_item.get('foldername')}. Expected list of 2 dicts. Got: {reasoning_data}"
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
            f"Error parsing 'reasoning' field for {dataset_item.get('foldername')}: {e}. Content: {dataset_item.get('reasoning')}"
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
    mllm_results: dict,  # {foldername: [output_perm0, output_perm1, ...]}
    config: dict,
    output_file_path: Path,
    progress_file_path: Path,
):
    global logger
    # ... (API setup for judge model, similar to RIA judge) ...
    judge_model_config = config["models"].get(judge_model_name)
    provider_name = judge_model_config["provider"]
    provider_config = config["api_providers"].get(provider_name)
    judge_api_key = get_api_key(judge_model_name)
    if not all([judge_model_config, provider_config, judge_api_key]):
        logger.error(f"Missing config/API key for judge model '{judge_model_name}'.")
        return
    judge_api_url = provider_config["base_url"] + judge_model_config["endpoint"]
    headers = {
        "Authorization": f"Bearer {judge_api_key}",
        "Content-Type": "application/json",
    }

    ica_judge_settings = config["evaluation_settings"]["ica"]["reasoning_judge"]
    system_prompt = ica_judge_settings["prompt"]
    group_size = ica_judge_settings["judge_group_size"]
    judge_max_tokens = (
        ica_judge_settings["judge_max_tokens_per_batch_multiplier"] * group_size
    )
    sleep_time = ica_judge_settings["sleep_time_after_judge_api"]
    expect_rotated = ica_judge_settings.get("expect_rotated_test_results", True)
    num_permutations_to_judge = 4 if expect_rotated else 1

    existing_judgements = (
        load_json(output_file_path) or {}
    )  # {foldername: [judgement_perm0, judgement_perm1, ...]}
    # Progress will track (foldername, permutation_index) pairs
    processed_item_perm_tuples = set(
        tuple(item) for item in (load_json(progress_file_path) or [])
    )

    logger.info(
        f"Loaded {sum(len(v) for v in existing_judgements.values() if isinstance(v, list))} existing judgement entries."
    )
    logger.info(
        f"Loaded {len(processed_item_perm_tuples)} processed (item, permutation_index) tuples for progress."
    )

    items_for_judge_api = (
        []
    )  # List of dicts: {foldername, perm_idx, dataset_item_ref, mllm_output_text}

    # Iterate safely, skipping items with image loading issues
    # safe_iterator = safe_dataset_iterator_with_error_info(dataset_ica_split, logger, desc="Preparing ICA items")
    # for original_idx, hf_item in safe_iterator: # Use original_idx if needed for anything

    for (
        hf_item
    ) in (
        dataset_ica_split
    ):  # Assuming direct iteration for now, add safe_iterator if needed
        foldername = hf_item.get("foldername")
        item_id = hf_item.get(
            "id"
        )  # Using 'id' as unique identifier for dataset item if 'foldername' isn't granular enough
        if not foldername:  # or not item_id
            logger.warning(
                f"Skipping dataset item due to missing 'foldername' or 'id': {hf_item}"
            )
            continue

        mllm_outputs_for_folder = mllm_results.get(foldername)
        if not mllm_outputs_for_folder or not isinstance(mllm_outputs_for_folder, list):
            logger.warning(f"No MLLM result list found for '{foldername}'. Skipping.")
            for perm_idx in range(
                num_permutations_to_judge
            ):  # Mark all perms as processed (with error)
                processed_item_perm_tuples.add((foldername, perm_idx))
            continue

        if len(mllm_outputs_for_folder) < num_permutations_to_judge:
            logger.warning(
                f"MLLM results for '{foldername}' has {len(mllm_outputs_for_folder)} items, expected {num_permutations_to_judge}. Some permutations might be skipped or marked as error."
            )

        if foldername not in existing_judgements:
            existing_judgements[foldername] = [None] * num_permutations_to_judge
        elif (
            len(existing_judgements[foldername]) < num_permutations_to_judge
        ):  # if loaded data is shorter
            existing_judgements[foldername].extend(
                [None]
                * (num_permutations_to_judge - len(existing_judgements[foldername]))
            )

        for perm_idx in range(num_permutations_to_judge):
            if (foldername, perm_idx) in processed_item_perm_tuples:
                # Optional: could re-validate existing_judgements[foldername][perm_idx] here
                continue  # Already processed

            if perm_idx >= len(mllm_outputs_for_folder):
                logger.warning(
                    f"Missing MLLM output for '{foldername}', permutation {perm_idx}. Marking as error."
                )
                existing_judgements[foldername][perm_idx] = {
                    "error": f"Missing MLLM output for permutation {perm_idx}."
                }
                processed_item_perm_tuples.add((foldername, perm_idx))
                continue

            mllm_output_text = mllm_outputs_for_folder[perm_idx]
            if (
                not isinstance(mllm_output_text, str)
                or "Error:" in mllm_output_text
                or not mllm_output_text.strip()
            ):
                logger.warning(
                    f"Invalid/error MLLM output for '{foldername}', perm {perm_idx}. Output: {str(mllm_output_text)[:100]}"
                )
                existing_judgements[foldername][perm_idx] = {
                    "error": "Invalid or error MLLM output for this permutation.",
                    "mllm_output_preview": str(mllm_output_text)[:100],
                }
                processed_item_perm_tuples.add((foldername, perm_idx))
                continue

            items_for_judge_api.append(
                {
                    "foldername": foldername,
                    "perm_idx": perm_idx,
                    "dataset_item_ref": hf_item,  # Keep a reference to the full HF dataset item
                    "mllm_output_text": mllm_output_text,
                }
            )

    if not items_for_judge_api:
        logger.info("No new items or permutations to judge for ICA.")
        save_json(
            list(sorted(list(processed_item_perm_tuples))), progress_file_path
        )  # Save progress
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
            current_batch_identifiers = []  # List of (foldername, perm_idx)

            for idx_in_batch, item_data in enumerate(batch_items_data):
                formatted_prompt_str = format_judge_prompt_item(
                    item_data["dataset_item_ref"],
                    item_data["mllm_output_text"],
                    item_data["perm_idx"],
                )
                if formatted_prompt_str is None:
                    logger.error(
                        f"Failed to format prompt for {item_data['foldername']}, perm {item_data['perm_idx']}. Skipping this item in batch."
                    )
                    # Mark as processed with error
                    existing_judgements[item_data["foldername"]][
                        item_data["perm_idx"]
                    ] = {"error": "Failed to format judge prompt."}
                    processed_item_perm_tuples.add(
                        (item_data["foldername"], item_data["perm_idx"])
                    )
                    continue

                user_prompts_for_batch_api.append(
                    f"{idx_in_batch + 1}.\n{formatted_prompt_str}"
                )
                current_batch_identifiers.append(
                    (item_data["foldername"], item_data["perm_idx"])
                )

            if not user_prompts_for_batch_api:  # All items in batch failed formatting
                pbar.update(len(batch_items_data))
                save_json(existing_judgements, output_file_path)  # Save errors
                save_json(
                    list(sorted(list(processed_item_perm_tuples))), progress_file_path
                )
                continue

            full_user_prompt = "\n\n".join(user_prompts_for_batch_api)
            payload = {
                "model": judge_model_name,
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
                )  # Longer timeout for ICA
                api_response.raise_for_status()
                response_json = api_response.json()
                judge_response_content = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )

                if not judge_response_content:  # Handle empty content
                    raise ValueError("Judge model returned no content.")

                if judge_response_content.startswith("```json"):
                    judge_response_content = judge_response_content[7:]
                if judge_response_content.endswith("```"):
                    judge_response_content = judge_response_content[:-3]
                parsed_judgements_batch = json.loads(judge_response_content.strip())

                for idx_in_api_batch, (foldername, perm_idx) in enumerate(
                    current_batch_identifiers
                ):
                    judgement_key = str(idx_in_api_batch + 1)
                    if judgement_key in parsed_judgements_batch:
                        single_judgement = parsed_judgements_batch[judgement_key]
                        if validate_ica_judgement(single_judgement):
                            existing_judgements[foldername][perm_idx] = single_judgement
                            logger.info(
                                f"Successfully judged: {foldername}, perm_idx {perm_idx}"
                            )
                        else:
                            logger.error(
                                f"Invalid judgement structure for {foldername}, perm {perm_idx}: {single_judgement}"
                            )
                            existing_judgements[foldername][perm_idx] = {
                                "error": "Invalid judgement structure from judge.",
                                "raw_judgement": single_judgement,
                            }
                    else:
                        logger.error(
                            f"Judge response missing key '{judgement_key}' for {foldername}, perm {perm_idx}."
                        )
                        existing_judgements[foldername][perm_idx] = {
                            "error": f"Judge response missing key '{judgement_key}'."
                        }
                    processed_item_perm_tuples.add((foldername, perm_idx))

            except (
                Exception
            ) as e:  # Catch requests errors, JSON errors, ValueErrors etc.
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
                    foldername,
                    perm_idx,
                ) in current_batch_identifiers:  # Mark all items in failed batch
                    if (
                        foldername,
                        perm_idx,
                    ) not in processed_item_perm_tuples:  # Avoid overwriting specific formatting error
                        existing_judgements[foldername][perm_idx] = {
                            "error": f"API/Processing error in batch: {str(e)[:100]}",
                            "raw_response_preview": raw_resp_text[:200],
                        }
                    processed_item_perm_tuples.add((foldername, perm_idx))
            finally:
                save_json(existing_judgements, output_file_path)
                save_json(
                    list(sorted(list(processed_item_perm_tuples))), progress_file_path
                )
                pbar.update(
                    len(batch_items_data)
                )  # Update for all items attempted in batch
                if i + group_size < len(items_for_judge_api):
                    time.sleep(sleep_time)

    logger.info(f"ICA Reasoning judgement finished for model: {model_to_judge_name}.")


def main():
    global logger
    parser = argparse.ArgumentParser(description="Perform ICA reasoning judgement.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="MLLM results to judge."
    )
    parser.add_argument(
        "--judge_model_name", type=str, default=None, help="LLM judge model."
    )
    args = parser.parse_args()

    model_to_judge_name = args.model_name

    try:
        config = get_config()
        judge_model_name_arg = (
            args.judge_model_name
            or config["evaluation_settings"]["ica"]["reasoning_judge"][
                "default_judge_model_name"
            ]
        )

        # Paths
        results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
        logs_dir = PROJECT_ROOT / config["general_settings"]["logs_base_dir"]
        progress_dir = PROJECT_ROOT / config["general_settings"]["progress_base_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        progress_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"ICA_{model_to_judge_name}_reasoning_judge.log"
        mllm_results_file = (
            results_dir / f"ICA_{model_to_judge_name}_results.json"
        )  # Input MLLM answers
        output_scoring_file = (
            results_dir / f"ICA_{model_to_judge_name}_reasoning_scoring.json"
        )  # Output scores
        progress_file = (
            progress_dir / f"ICA_{model_to_judge_name}_reasoning_judge_progress.json"
        )

        logger = setup_logger(f"ICA_ReasoningJudge_{model_to_judge_name}", log_file)
        logger.info(
            f"Starting ICA reasoning judgement for MLLM: {model_to_judge_name} using Judge: {judge_model_name_arg}"
        )
        # ... (log other paths)

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
        Path(hf_cache).mkdir(parents=True, exist_ok=True)

        # Specify columns to load to avoid image decoding issues, if possible
        # For ICA, we need 'foldername', 'id', 'description1'-'description4', 'relation', 'reasoning'
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
                # Decide if you want to proceed with all columns or exit
                # For now, let's try to proceed with all, but it might hit image errors
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
            progress_file,
        )

    except Exception as e:
        if logger:
            logger.critical(
                f"Critical error in ICA_reasoning_judge main: {e}", exc_info=True
            )
        else:
            print(f"Critical error: {e}")


if __name__ == "__main__":
    # Run from MM-OPERA root:
    # python -m evaluation.ICA.ICA_reasoning_judge --model_name gpt-4o
    main()
