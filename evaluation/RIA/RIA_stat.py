import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from datasets import load_dataset, config
from tqdm import tqdm

from evaluation.utils import PROJECT_ROOT, load_json, save_json
from evaluation.config_loader import get_config


# Hyperparameters
DECAY_FACTOR = 0.9
ALPHA = 0.9


def load_dataset_info():
    """Load the dataset information"""
    try:
        config.HF_DATASETS_CACHE = "./dataset"
        columns_to_load = [
            "id",
            "foldername",
            "relation",
            "domain",
            "type",
            "culture",
            "language",
            "explanation",
            "hop_count",
            "reasoning",
            "perception",
            "conception",
        ]
        ds = load_dataset("titic/MM-OPERA", keep_in_memory=False)
        ria_dataset = ds["ria"].select_columns(columns_to_load)
        return ria_dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def analyze_regular_judge(model_name, ria_dataset):
    """Analyze the results of regular judge"""
    config = get_config()
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]

    # Load the regular judge results
    judge_file_path = results_dir / f"{model_name}_RIA_regular_scoring.json"
    if not judge_file_path.exists():
        print(f"Regular judge file does not exist: {judge_file_path}")
        return None

    judge_results = load_json(judge_file_path)
    if not judge_results:
        print(f"Regular judge file is empty or has incorrect format: {judge_file_path}")
        return None

    # Initialize the statistics data
    stats = {
        "test_num": len(judge_results),
        "score_rate": 0,
        "high_score_rate": {"HR-3": 0, "HR-4": 0, "delta_HR": 0},
        "score_distribution": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
        "domain": defaultdict(lambda: {"count": 0, "total_score": 0}),
        "culture": defaultdict(lambda: {"count": 0, "total_score": 0}),
        "type": defaultdict(lambda: {"count": 0, "total_score": 0}),
        "language": defaultdict(lambda: {"count": 0, "total_score": 0}),
        "hop_count": defaultdict(lambda: {"count": 0, "total_score": 0}),
        "ability_dimension_L2": {
            "recognition": {"count": 0, "total_score": 0},
            "context": {"count": 0, "total_score": 0},
            "interaction": {"count": 0, "total_score": 0},
            "logic": {"count": 0, "total_score": 0},
            "semantic": {"count": 0, "total_score": 0},
            "reasoning": {"count": 0, "total_score": 0},
        },
        "ability_dimension_L3": {
            "Visual Similarity": {"count": 0, "total_score": 0},
            "Semantic Object": {"count": 0, "total_score": 0},
            "Social Insight": {"count": 0, "total_score": 0},
            "Relational Perception": {"count": 0, "total_score": 0},
            "Contextual Sensory Cues": {"count": 0, "total_score": 0},
            "Scene Contextualization": {"count": 0, "total_score": 0},
            "Abstract Interpretation": {"count": 0, "total_score": 0},
            "Functional Links": {"count": 0, "total_score": 0},
            "Causal Connections": {"count": 0, "total_score": 0},
            "Thematic Links": {"count": 0, "total_score": 0},
            "Cultural Reference": {"count": 0, "total_score": 0},
            "Hierarchical Association": {"count": 0, "total_score": 0},
            "Analogical Reasoning": {"count": 0, "total_score": 0},
        },
    }

    # Create a mapping from ID to dataset items
    id_to_item = {item["id"]: item for item in ria_dataset}

    # Count the scores
    total_score = 0
    hr3_count = 0
    hr4_count = 0

    for item_id, result in judge_results.items():
        try:
            if not isinstance(result, dict) or "score_judge" not in result:
                print(
                    f"Warning: Project {item_id} missing score_judge field or incorrect format, skipping"
                )
                continue

            # Convert the score to an integer, handle possible format errors
            try:
                score = int(result["score_judge"])
                if not (0 <= score <= 4):
                    print(
                        f"Warning: Project {item_id} score {score} out of range (0-4), skipping"
                    )
                    continue
            except (ValueError, TypeError):
                print(
                    f"Warning: Project {item_id} score format is incorrect: {result['score_judge']}, skipping"
                )
                continue

            total_score += score

            # Update the score distribution
            stats["score_distribution"][str(score)] += 1

            # Update the high score rate
            if score >= 3:
                hr3_count += 1
            if score == 4:
                hr4_count += 1

            # Get the dataset item
            if item_id in id_to_item:
                item = id_to_item[item_id]

                # Update the statistics of each dimension
                domain = item.get("domain", "unknown")
                stats["domain"][domain]["count"] += 1
                stats["domain"][domain]["total_score"] += score

                culture = item.get("culture", "unknown")
                stats["culture"][culture]["count"] += 1
                stats["culture"][culture]["total_score"] += score

                type_val = item.get("type", "unknown")
                stats["type"][type_val]["count"] += 1
                stats["type"][type_val]["total_score"] += score

                language = item.get("language", "unknown")
                stats["language"][language]["count"] += 1
                stats["language"][language]["total_score"] += score

                hop_count = item.get("hop_count", "unknown")
                stats["hop_count"][str(hop_count)]["count"] += 1
                stats["hop_count"][str(hop_count)]["total_score"] += score

                # Process the ability dimension
                perception = item.get("perception", "")
                conception = item.get("conception", "")

                # Process L2 and L3 dimensions
                if perception:
                    # L3 dimensions for perception
                    if "Visual Similarity" in perception:
                        stats["ability_dimension_L3"]["Visual Similarity"]["count"] += 1
                        stats["ability_dimension_L3"]["Visual Similarity"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["recognition"]["count"] += 1
                        stats["ability_dimension_L2"]["recognition"][
                            "total_score"
                        ] += score

                    if "Semantic Object" in perception:
                        stats["ability_dimension_L3"]["Semantic Object"]["count"] += 1
                        stats["ability_dimension_L3"]["Semantic Object"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["recognition"]["count"] += 1
                        stats["ability_dimension_L2"]["recognition"][
                            "total_score"
                        ] += score

                    if "Social Insight" in perception:
                        stats["ability_dimension_L3"]["Social Insight"]["count"] += 1
                        stats["ability_dimension_L3"]["Social Insight"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["interaction"]["count"] += 1
                        stats["ability_dimension_L2"]["interaction"][
                            "total_score"
                        ] += score

                    if "Relational Perception" in perception:
                        stats["ability_dimension_L3"]["Relational Perception"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Relational Perception"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["interaction"]["count"] += 1
                        stats["ability_dimension_L2"]["interaction"][
                            "total_score"
                        ] += score

                    if "Contextual Sensory Cues" in perception:
                        stats["ability_dimension_L3"]["Contextual Sensory Cues"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Contextual Sensory Cues"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["context"]["count"] += 1
                        stats["ability_dimension_L2"]["context"]["total_score"] += score

                    if "Scene Contextualization" in perception:
                        stats["ability_dimension_L3"]["Scene Contextualization"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Scene Contextualization"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["context"]["count"] += 1
                        stats["ability_dimension_L2"]["context"]["total_score"] += score

                    if "Abstract Interpretation" in perception:
                        stats["ability_dimension_L3"]["Abstract Interpretation"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Abstract Interpretation"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["context"]["count"] += 1
                        stats["ability_dimension_L2"]["context"]["total_score"] += score

                if conception:
                    # L3 dimensions for conception
                    if "Functional Links" in conception:
                        stats["ability_dimension_L3"]["Functional Links"]["count"] += 1
                        stats["ability_dimension_L3"]["Functional Links"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["logic"]["count"] += 1
                        stats["ability_dimension_L2"]["logic"]["total_score"] += score

                    if "Causal Connections" in conception:
                        stats["ability_dimension_L3"]["Causal Connections"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Causal Connections"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["logic"]["count"] += 1
                        stats["ability_dimension_L2"]["logic"]["total_score"] += score

                    if "Thematic Links" in conception:
                        stats["ability_dimension_L3"]["Thematic Links"]["count"] += 1
                        stats["ability_dimension_L3"]["Thematic Links"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["semantic"]["count"] += 1
                        stats["ability_dimension_L2"]["semantic"][
                            "total_score"
                        ] += score

                    if "Cultural Reference" in conception:
                        stats["ability_dimension_L3"]["Cultural Reference"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Cultural Reference"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["semantic"]["count"] += 1
                        stats["ability_dimension_L2"]["semantic"][
                            "total_score"
                        ] += score

                    if "Hierarchical Association" in conception:
                        stats["ability_dimension_L3"]["Hierarchical Association"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Hierarchical Association"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["reasoning"]["count"] += 1
                        stats["ability_dimension_L2"]["reasoning"][
                            "total_score"
                        ] += score

                    if "Analogical Reasoning" in conception:
                        stats["ability_dimension_L3"]["Analogical Reasoning"][
                            "count"
                        ] += 1
                        stats["ability_dimension_L3"]["Analogical Reasoning"][
                            "total_score"
                        ] += score
                        stats["ability_dimension_L2"]["reasoning"]["count"] += 1
                        stats["ability_dimension_L2"]["reasoning"][
                            "total_score"
                        ] += score

        except Exception as e:
            print(f"Error occurred while processing project {item_id}: {e}, skipping")
            continue

    # Calculate the score rate
    if stats["test_num"] > 0:
        stats["score_rate"] = total_score / (4 * stats["test_num"])
        stats["high_score_rate"]["HR-3"] = hr3_count / stats["test_num"]
        stats["high_score_rate"]["HR-4"] = hr4_count / stats["test_num"]
        stats["high_score_rate"]["delta_HR"] = (
            stats["high_score_rate"]["HR-3"] - stats["high_score_rate"]["HR-4"]
        )

        # Calculate the distribution percentage
        for key in stats["score_distribution"]:
            stats["score_distribution"][key] = (
                stats["score_distribution"][key] / stats["test_num"]
            )

    # Convert the defaultdict to dict for JSON serialization
    stats["domain"] = {
        k: {
            "count": v["count"],
            "score_rate": v["total_score"] / (4 * v["count"]) if v["count"] > 0 else 0,
        }
        for k, v in stats["domain"].items()
    }
    stats["culture"] = {
        k: {
            "count": v["count"],
            "score_rate": v["total_score"] / (4 * v["count"]) if v["count"] > 0 else 0,
        }
        for k, v in stats["culture"].items()
    }
    stats["type"] = {
        k: {
            "count": v["count"],
            "score_rate": v["total_score"] / (4 * v["count"]) if v["count"] > 0 else 0,
        }
        for k, v in stats["type"].items()
    }
    stats["language"] = {
        k: {
            "count": v["count"],
            "score_rate": v["total_score"] / (4 * v["count"]) if v["count"] > 0 else 0,
        }
        for k, v in stats["language"].items()
    }
    stats["hop_count"] = {
        k: {
            "count": v["count"],
            "score_rate": v["total_score"] / (4 * v["count"]) if v["count"] > 0 else 0,
        }
        for k, v in stats["hop_count"].items()
    }

    # Convert L2 dimensions
    for key in stats["ability_dimension_L2"]:
        count = stats["ability_dimension_L2"][key]["count"]
        total_score = stats["ability_dimension_L2"][key]["total_score"]
        if count > 0:
            stats["ability_dimension_L2"][key] = {
                "count": count,
                "score_rate": total_score / (4 * count),
            }
        else:
            stats["ability_dimension_L2"][key] = {"count": 0, "score_rate": 0}

    # Convert L3 dimensions
    for key in stats["ability_dimension_L3"]:
        count = stats["ability_dimension_L3"][key]["count"]
        total_score = stats["ability_dimension_L3"][key]["total_score"]
        if count > 0:
            stats["ability_dimension_L3"][key] = {
                "count": count,
                "score_rate": total_score / (4 * count),
            }
        else:
            stats["ability_dimension_L3"][key] = {"count": 0, "score_rate": 0}

    return stats


def analyze_reasoning_judge(model_name, ria_dataset):
    """Analyze the results of reasoning judge"""
    config = get_config()
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]

    # Load the reasoning judge results
    reasoning_file_path = results_dir / f"{model_name}_RIA_reasoning_scoring.json"
    if not reasoning_file_path.exists():
        print(f"Reasoning judge file does not exist: {reasoning_file_path}")
        return None

    reasoning_results = load_json(reasoning_file_path)
    if not reasoning_results:
        print(
            f"Reasoning judge file is empty or has incorrect format: {reasoning_file_path}"
        )
        return None

    # Initialize the statistics data
    stats = {
        "test_num": len(reasoning_results),
        "decay_factor": DECAY_FACTOR,
        "alpha": ALPHA,
        "avg_reasoning_score": 0,
        "reasoning_score_distribution": {"0-1": 0, "1-2": 0, "2-3": 0},
        "hop_count_distribution": defaultdict(int),
        "reasonableness_distribution": {
            "0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1": 0,
        },
        "distinctiveness_distribution": {
            "0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1": 0,
        },
        "knowledgeability_distribution": {"0-0.5": 0, "0.5-1": 0},
        "avg_reasonableness": 0,
        "avg_distinctiveness": 0,
        "avg_knowledgeability": 0,
    }

    total_reasoning_score = 0
    total_reasonableness = 0
    total_distinctiveness = 0
    total_knowledgeability = 0
    total_hops = 0
    valid_items = 0

    for item_id, result in reasoning_results.items():
        try:
            if not isinstance(result, dict) or "hop_quality" not in result:
                print(
                    f"Warning: Project {item_id} missing hop_quality field or incorrect format, skipping"
                )
                continue

            hop_quality = result["hop_quality"]
            if not isinstance(hop_quality, dict) or not hop_quality:
                print(
                    f"Warning: Project {item_id} hop_quality is empty or incorrect format, skipping"
                )
                continue

            hop_count = len(hop_quality)
            stats["hop_count_distribution"][str(hop_count)] += 1

            # Calculate the reasoning score
            score_list = []
            valid_hop = True

            for hop, hop_score_list in hop_quality.items():
                try:
                    # Verify the format of hop_score_list
                    if not isinstance(hop_score_list, list) or len(hop_score_list) != 3:
                        print(
                            f"Warning: Project {item_id} hop {hop} score format is incorrect, skipping"
                        )
                        valid_hop = False
                        continue

                    # Ensure all scores are numeric
                    reasonableness = float(hop_score_list[0])
                    distinctiveness = float(hop_score_list[1])
                    knowledgeability = float(hop_score_list[2])

                    # Ensure the scores are within the valid range
                    if not (0 <= reasonableness <= 1):
                        print(
                            f"Warning: Project {item_id} hop {hop} reasonableness value {reasonableness} out of range [0,1], adjusted to the range"
                        )
                        reasonableness = max(0, min(1, reasonableness))

                    if not (0 <= distinctiveness <= 1):
                        print(
                            f"Warning: Project {item_id} hop {hop} distinctiveness value {distinctiveness} out of range [0,1], adjusted to the range"
                        )
                        distinctiveness = max(0, min(1, distinctiveness))

                    # Ensure knowledgeability is 0 or 1
                    if knowledgeability != 0 and knowledgeability != 1:
                        print(
                            f"Warning: Project {item_id} hop {hop} knowledgeability value {knowledgeability} is not 0 or 1, adjusted to the nearest value"
                        )
                        knowledgeability = 0 if knowledgeability < 0.5 else 1

                    # Calculate the hop score
                    hop_score = min(
                        1,
                        ALPHA * reasonableness * distinctiveness
                        + (1 - ALPHA) * knowledgeability,
                    )
                    score_list.append(hop_score)

                    # Count the distribution of each dimension
                    total_reasonableness += reasonableness
                    total_distinctiveness += distinctiveness
                    total_knowledgeability += knowledgeability
                    total_hops += 1

                    # Update the reasonableness distribution
                    if 0 <= reasonableness < 0.2:
                        stats["reasonableness_distribution"]["0-0.2"] += 1
                    elif 0.2 <= reasonableness < 0.4:
                        stats["reasonableness_distribution"]["0.2-0.4"] += 1
                    elif 0.4 <= reasonableness < 0.6:
                        stats["reasonableness_distribution"]["0.4-0.6"] += 1
                    elif 0.6 <= reasonableness < 0.8:
                        stats["reasonableness_distribution"]["0.6-0.8"] += 1
                    else:
                        stats["reasonableness_distribution"]["0.8-1"] += 1

                    # Update the distinctiveness distribution
                    if 0 <= distinctiveness < 0.2:
                        stats["distinctiveness_distribution"]["0-0.2"] += 1
                    elif 0.2 <= distinctiveness < 0.4:
                        stats["distinctiveness_distribution"]["0.2-0.4"] += 1
                    elif 0.4 <= distinctiveness < 0.6:
                        stats["distinctiveness_distribution"]["0.4-0.6"] += 1
                    elif 0.6 <= distinctiveness < 0.8:
                        stats["distinctiveness_distribution"]["0.6-0.8"] += 1
                    else:
                        stats["distinctiveness_distribution"]["0.8-1"] += 1

                    # Update the knowledgeability distribution
                    if 0 <= knowledgeability < 0.5:
                        stats["knowledgeability_distribution"]["0-0.5"] += 1
                    else:
                        stats["knowledgeability_distribution"]["0.5-1"] += 1
                except (ValueError, TypeError, IndexError) as e:
                    print(
                        f"Error occurred while processing project {item_id} hop {hop}: {e}, skipping"
                    )
                    valid_hop = False
                    continue

            if not valid_hop or not score_list:
                print(f"Warning: Project {item_id} has no valid hop scores, skipping")
                continue

            # Calculate the weighted sum
            score = 0
            for i, hop_score in enumerate(score_list):
                score += hop_score * (DECAY_FACTOR**i)

            total_reasoning_score += score
            valid_items += 1

            # Update the reasoning score distribution
            if 0 <= score < 1:
                stats["reasoning_score_distribution"]["0-1"] += 1
            elif 1 <= score < 2:
                stats["reasoning_score_distribution"]["1-2"] += 1
            else:
                stats["reasoning_score_distribution"]["2-3"] += 1
        except Exception as e:
            print(f"Error occurred while processing project {item_id}: {e}, skipping")
            continue

    # Convert distributions from counts to percentages
    if valid_items > 0:
        stats["test_num"] = valid_items  # Update to the actual number of valid tests
        stats["avg_reasoning_score"] = total_reasoning_score / valid_items

        # Convert reasoning_score_distribution to percentages
        total_reasoning = sum(stats["reasoning_score_distribution"].values())
        if total_reasoning > 0:
            for key in stats["reasoning_score_distribution"]:
                stats["reasoning_score_distribution"][key] = (
                    stats["reasoning_score_distribution"][key] / total_reasoning
                )

        # Convert hop_count_distribution to percentages
        total_hops_count = sum(stats["hop_count_distribution"].values())
        if total_hops_count > 0:
            for key in stats["hop_count_distribution"]:
                stats["hop_count_distribution"][key] = (
                    stats["hop_count_distribution"][key] / total_hops_count
                )

    if total_hops > 0:
        stats["avg_reasonableness"] = total_reasonableness / total_hops
        stats["avg_distinctiveness"] = total_distinctiveness / total_hops
        stats["avg_knowledgeability"] = total_knowledgeability / total_hops

        # Convert reasonableness_distribution to percentages
        total_reasonableness_count = sum(stats["reasonableness_distribution"].values())
        if total_reasonableness_count > 0:
            for key in stats["reasonableness_distribution"]:
                stats["reasonableness_distribution"][key] = (
                    stats["reasonableness_distribution"][key]
                    / total_reasonableness_count
                )

        # Convert distinctiveness_distribution to percentages
        total_distinctiveness_count = sum(
            stats["distinctiveness_distribution"].values()
        )
        if total_distinctiveness_count > 0:
            for key in stats["distinctiveness_distribution"]:
                stats["distinctiveness_distribution"][key] = (
                    stats["distinctiveness_distribution"][key]
                    / total_distinctiveness_count
                )

        # Convert knowledgeability_distribution to percentages
        total_knowledgeability_count = sum(
            stats["knowledgeability_distribution"].values()
        )
        if total_knowledgeability_count > 0:
            for key in stats["knowledgeability_distribution"]:
                stats["knowledgeability_distribution"][key] = (
                    stats["knowledgeability_distribution"][key]
                    / total_knowledgeability_count
                )

    # Convert the defaultdict to a normal dict for JSON serialization
    stats["hop_count_distribution"] = dict(stats["hop_count_distribution"])

    return stats


def main(model_name=None):
    """Main function"""
    if not model_name:
        config = get_config()
        model_name = config["evaluation_settings"]["ria"]["default_model_name"]

    print(f"Start analyzing the RIA results of {model_name}...")

    # Load the dataset
    ria_dataset = load_dataset_info()
    if not ria_dataset:
        print("Failed to load the dataset, exiting the program")
        return

    # Analyze the results of regular judge
    regular_stats = analyze_regular_judge(model_name, ria_dataset)

    # Analyze the results of reasoning judge
    reasoning_stats = analyze_reasoning_judge(model_name, ria_dataset)

    # Merge the results
    stats = {
        "model_name": model_name,
        "regular_judge": (
            regular_stats if regular_stats else "No regular judge results found"
        ),
        "reasoning_judge": (
            reasoning_stats if reasoning_stats else "No reasoning judge results found"
        ),
    }

    # Save the results
    config = get_config()
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
    results_file_path = results_dir / f"{model_name}_RIA_stats.json"

    save_json(stats, results_file_path)
    print(f"Statistics saved to {results_file_path}")
    # Print the summary
    print(f"Model: {model_name}")
    if regular_stats:
        print(
            f"Regular Judge - Test Count: {regular_stats['test_num']}, Score Rate: {regular_stats['score_rate']:.2f}"
        )
        print(
            f"High Score Rate (≥3): {regular_stats['high_score_rate']['HR-3']:.2f}, (=4): {regular_stats['high_score_rate']['HR-4']:.2f}"
        )
    else:
        print("Regular Judge: Not available")

    if reasoning_stats:
        print(
            f"Reasoning Judge - Test Count: {reasoning_stats['test_num']}, Avg Score: {reasoning_stats['avg_reasoning_score']:.2f}"
        )
    else:
        print("Reasoning Judge: Not available")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the results of RIA")
    parser.add_argument("--model_name", type=str, help="The model name to be analyzed")

    args = parser.parse_args()
    main(args.model_name)
