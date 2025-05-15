import json
import os
import logging
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np

from evaluation.utils import load_json, save_json, PROJECT_ROOT
from evaluation.config_loader import get_config

# Constant definitions
DECAY_FACTOR = 0.9
ALPHA = 0.9


def load_dataset_info(config):
    """Load the dataset information, get domain, culture, type, etc."""
    try:
        from datasets import load_dataset
        from datasets import config as ds_config

        # Set the HF cache directory
        ds_config.HF_DATASETS_CACHE = config["general_settings"].get(
            "huggingface_cache_dir", "./dataset"
        )
        dataset_name = config["general_settings"].get(
            "huggingface_dataset_name", "titic/MM-OPERA"
        )

        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset(dataset_name)

        # Create a mapping from ID to metadata
        metadata_map = {}
        for item in ds["ica"]:
            item_id = item.get("id")
            if item_id:
                metadata_map[item_id] = {
                    "domain": item.get("domain"),
                    "culture": item.get("culture"),
                    "type": item.get("type"),
                    "language": item.get("language"),
                    "hop_count": item.get("hop_count"),
                    "perception": item.get("perception", []),
                    "conception": item.get("conception", []),
                    "reasoning": item.get("reasoning"),
                }

        print(f"Loaded metadata for {len(metadata_map)} items")
        return metadata_map
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}


def parse_perception_conception(metadata_map):
    """Parse the L-3 labels of perception and conception, calculate the L-2 labels"""
    for item_id, metadata in metadata_map.items():
        reasoning_data = metadata.get("reasoning")
        if not reasoning_data:
            continue

        try:
            if isinstance(reasoning_data, str):
                reasoning_data = json.loads(reasoning_data)

            perception_L3 = defaultdict(list)
            conception_L3 = defaultdict(list)

            for pair in reasoning_data:
                perception_list = pair.get("perception", [])
                conception_list = pair.get("conception", [])

                for p in perception_list:
                    perception_L3[p].append(1)

                for c in conception_list:
                    conception_L3[c].append(1)

            # Calculate the average value of each L3 label
            for key in perception_L3:
                perception_L3[key] = sum(perception_L3[key]) / len(perception_L3[key])

            for key in conception_L3:
                conception_L3[key] = sum(conception_L3[key]) / len(conception_L3[key])

            # Calculate the L2 labels
            recognition = (
                perception_L3.get("Visual Similarity", 0)
                + perception_L3.get("Semantic Object", 0)
            ) / 2
            context = (
                perception_L3.get("Contextual Sensory Cues", 0)
                + perception_L3.get("Scene Contextualization", 0)
                + perception_L3.get("Abstract Interpretation", 0)
            ) / 3
            interaction = (
                perception_L3.get("Social Insight", 0)
                + perception_L3.get("Relational Perception", 0)
            ) / 2
            logic = (
                conception_L3.get("Functional Links", 0)
                + conception_L3.get("Causal Connections", 0)
            ) / 2
            semantic = (
                conception_L3.get("Thematic Links", 0)
                + conception_L3.get("Cultural Reference", 0)
            ) / 2
            reasoning = (
                conception_L3.get("Hierarchical Association", 0)
                + conception_L3.get("Analogical Reasoning", 0)
            ) / 2

            # Store the L2 labels
            metadata["perception_L2"] = {
                "recognition": recognition,
                "context": context,
                "interaction": interaction,
            }

            metadata["conception_L2"] = {
                "logic": logic,
                "semantic": semantic,
                "reasoning": reasoning,
            }

        except Exception as e:
            print(f"Error parsing perception/conception for {item_id}: {e}")

    return metadata_map


def analyze_regular_judge(model_name, config, metadata_map):
    """Analyze the results of the regular judge"""
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
    judge_scores_file = results_dir / f"{model_name}_ICA_regular_scoring.json"

    if not judge_scores_file.exists():
        print(f"Regular judge scores file not found: {judge_scores_file}")
        return None

    print(f"Analyzing regular judge scores from {judge_scores_file}")
    judge_scores = load_json(judge_scores_file)

    if not judge_scores:
        print("Regular judge scores file is empty or invalid")
        return None

    # Initialize the statistical results
    stats = {
        "test_num": 0,
        "score_rate": 0,
        "high_score_rate": {"HR-3": 0, "HR-4": 0, "delta_HR": 0},
        "score_distribution": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
        "domain": {},
        "culture": {},
        "type": {},
        "language": {},
        "hop_count": {},
        "ability_dimension_L2": {
            "recognition": {"count": 0, "score_rate": 0},
            "context": {"count": 0, "score_rate": 0},
            "interaction": {"count": 0, "score_rate": 0},
            "logic": {"count": 0, "score_rate": 0},
            "semantic": {"count": 0, "score_rate": 0},
            "reasoning": {"count": 0, "score_rate": 0},
        },
        "ability_dimension_L3": {
            "Visual Similarity": {"count": 0, "score_rate": 0},
            "Semantic Object": {"count": 0, "score_rate": 0},
            "Social Insight": {"count": 0, "score_rate": 0},
            "Relational Perception": {"count": 0, "score_rate": 0},
            "Contextual Sensory Cues": {"count": 0, "score_rate": 0},
            "Scene Contextualization": {"count": 0, "score_rate": 0},
            "Abstract Interpretation": {"count": 0, "score_rate": 0},
            "Functional Links": {"count": 0, "score_rate": 0},
            "Causal Connections": {"count": 0, "score_rate": 0},
            "Thematic Links": {"count": 0, "score_rate": 0},
            "Cultural Reference": {"count": 0, "score_rate": 0},
            "Hierarchical Association": {"count": 0, "score_rate": 0},
            "Analogical Reasoning": {"count": 0, "score_rate": 0},
        },
    }

    domain_scores = defaultdict(list)
    culture_scores = defaultdict(list)
    type_scores = defaultdict(list)
    language_scores = defaultdict(list)
    hop_count_scores = defaultdict(list)

    # L2 dimension scores
    recognition_scores = []
    context_scores = []
    interaction_scores = []
    logic_scores = []
    semantic_scores = []
    reasoning_scores = []

    # L3 dimension scores
    l3_scores = {
        "Visual Similarity": [],
        "Semantic Object": [],
        "Social Insight": [],
        "Relational Perception": [],
        "Contextual Sensory Cues": [],
        "Scene Contextualization": [],
        "Abstract Interpretation": [],
        "Functional Links": [],
        "Causal Connections": [],
        "Thematic Links": [],
        "Cultural Reference": [],
        "Hierarchical Association": [],
        "Analogical Reasoning": [],
    }

    # Define L3 to L2 mapping
    l3_to_l2 = {
        "Visual Similarity": "recognition",
        "Semantic Object": "recognition",
        "Contextual Sensory Cues": "context",
        "Scene Contextualization": "context",
        "Abstract Interpretation": "context",
        "Social Insight": "interaction",
        "Relational Perception": "interaction",
        "Functional Links": "logic",
        "Causal Connections": "logic",
        "Thematic Links": "semantic",
        "Cultural Reference": "semantic",
        "Hierarchical Association": "reasoning",
        "Analogical Reasoning": "reasoning",
    }

    total_score = 0
    hr3_count = 0
    hr4_count = 0

    # Iterate over each test item
    for item_id, item_scores in judge_scores.items():
        if not isinstance(item_scores, list):
            print(
                f"Invalid scores format for item {item_id}, expected list but got {type(item_scores)}"
            )
            continue

        metadata = metadata_map.get(item_id)
        if not metadata:
            print(f"Metadata not found for item {item_id}")
            continue

        # Process each sub-test
        for sub_idx, sub_score in enumerate(item_scores):
            try:
                if not isinstance(sub_score, dict):
                    print(
                        f"Invalid sub-score format for item {item_id}[{sub_idx}], expected dict but got {type(sub_score)}"
                    )
                    continue

                score_str = sub_score.get("score_judge", "").strip()
                if not score_str or not score_str.isdigit():
                    print(
                        f"Invalid score value for item {item_id}[{sub_idx}]: {score_str}"
                    )
                    continue

                score = int(score_str)
                if score < 0 or score > 4:
                    print(f"Score out of range for item {item_id}[{sub_idx}]: {score}")
                    continue

                # Update the statistical information
                stats["test_num"] += 1
                total_score += score
                stats["score_distribution"][str(score)] += 1

                if score >= 3:
                    hr3_count += 1
                if score == 4:
                    hr4_count += 1

                # Count by dimension
                domain = metadata.get("domain", "unknown")
                culture = metadata.get("culture", "unknown")
                type_val = metadata.get("type", "unknown")
                language = metadata.get("language", "unknown")
                hop_count = metadata.get("hop_count", "unknown")

                # Handle multiple domains
                if isinstance(domain, str) and "," in domain:
                    domains = [d.strip() for d in domain.split(",")]
                    for d in domains:
                        domain_scores[d].append(score)
                else:
                    domain_scores[domain].append(score)

                culture_scores[culture].append(score)
                type_scores[type_val].append(score)
                language_scores[language].append(score)
                hop_count_scores[str(hop_count)].append(score)

                # Get reasoning data from metadata
                reasoning_data = metadata.get("reasoning")
                if reasoning_data:
                    # Parse reasoning data if it's a string
                    if isinstance(reasoning_data, str):
                        try:
                            reasoning_data = json.loads(reasoning_data)
                        except json.JSONDecodeError:
                            print(f"Failed to parse reasoning data for {item_id}")
                            reasoning_data = []

                    # Process each reasoning pair
                    for pair in reasoning_data:
                        # Extract perception and conception labels
                        perception_labels = pair.get("perception", [])
                        conception_labels = pair.get("conception", [])

                        # Process perception L3 labels
                        for label in perception_labels:
                            if label in l3_scores:
                                l3_scores[label].append(score)

                                # Map to L2 category
                                if label in l3_to_l2:
                                    l2_category = l3_to_l2[label]
                                    if l2_category == "recognition":
                                        recognition_scores.append(score)
                                    elif l2_category == "context":
                                        context_scores.append(score)
                                    elif l2_category == "interaction":
                                        interaction_scores.append(score)

                        # Process conception L3 labels
                        for label in conception_labels:
                            if label in l3_scores:
                                l3_scores[label].append(score)

                                # Map to L2 category
                                if label in l3_to_l2:
                                    l2_category = l3_to_l2[label]
                                    if l2_category == "logic":
                                        logic_scores.append(score)
                                    elif l2_category == "semantic":
                                        semantic_scores.append(score)
                                    elif l2_category == "reasoning":
                                        reasoning_scores.append(score)

            except Exception as e:
                print(f"Error processing score for item {item_id}[{sub_idx}]: {e}")
                continue

    # Calculate the statistical results
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

        # Process domain scores
        for domain, scores in domain_scores.items():
            if scores:
                stats["domain"][domain] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

        # Process culture scores
        for culture, scores in culture_scores.items():
            if scores:
                stats["culture"][culture] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

        # Process type scores
        for type_val, scores in type_scores.items():
            if scores:
                stats["type"][type_val] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

        # Process language scores
        for language, scores in language_scores.items():
            if scores:
                stats["language"][language] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

        # Process hop count scores
        for hop_count, scores in hop_count_scores.items():
            if scores:
                stats["hop_count"][hop_count] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

        # Process L2 dimension scores
        if recognition_scores:
            stats["ability_dimension_L2"]["recognition"] = {
                "count": len(recognition_scores),
                "score_rate": sum(recognition_scores) / (4 * len(recognition_scores)),
            }

        if context_scores:
            stats["ability_dimension_L2"]["context"] = {
                "count": len(context_scores),
                "score_rate": sum(context_scores) / (4 * len(context_scores)),
            }

        if interaction_scores:
            stats["ability_dimension_L2"]["interaction"] = {
                "count": len(interaction_scores),
                "score_rate": sum(interaction_scores) / (4 * len(interaction_scores)),
            }

        if logic_scores:
            stats["ability_dimension_L2"]["logic"] = {
                "count": len(logic_scores),
                "score_rate": sum(logic_scores) / (4 * len(logic_scores)),
            }

        if semantic_scores:
            stats["ability_dimension_L2"]["semantic"] = {
                "count": len(semantic_scores),
                "score_rate": sum(semantic_scores) / (4 * len(semantic_scores)),
            }

        if reasoning_scores:
            stats["ability_dimension_L2"]["reasoning"] = {
                "count": len(reasoning_scores),
                "score_rate": sum(reasoning_scores) / (4 * len(reasoning_scores)),
            }

        # Process L3 dimension scores
        for label, scores in l3_scores.items():
            if scores:
                stats["ability_dimension_L3"][label] = {
                    "count": len(scores),
                    "score_rate": sum(scores) / (4 * len(scores)),
                }

    return stats


def calculate_reasoning_score(hop_quality):
    """计算reasoning score"""
    try:
        score_list = []
        for hop, hop_score_list in hop_quality.items():
            # Ensure hop_score_list is a valid list and contains 3 elements
            if not isinstance(hop_score_list, list) or len(hop_score_list) != 3:
                print(f"Invalid hop score format for {hop}: {hop_score_list}")
                continue

            # Ensure K value is 0 or 1
            if hop_score_list[2] != 0 and hop_score_list[2] != 1:
                print(
                    f"Invalid K value in hop {hop}: {hop_score_list[2]}, normalizing to 0 or 1"
                )
                hop_score_list[2] = 0 if hop_score_list[2] < 0.5 else 1

            # Ensure R and P values are between 0 and 1
            r_value = max(0, min(1, hop_score_list[0]))
            p_value = max(0, min(1, hop_score_list[1]))
            k_value = hop_score_list[2]

            # Calculate the hop score
            hop_score = min(1, ALPHA * r_value * p_value + (1 - ALPHA) * k_value)
            score_list.append(hop_score)

        # Calculate the weighted score
        score = 0
        for i, hop_score in enumerate(score_list):
            score += hop_score * (DECAY_FACTOR**i)

        return score
    except Exception as e:
        print(f"Error calculating reasoning score: {e}")
        return 0


def analyze_reasoning_judge(model_name, config):
    """Analyze the results of the reasoning judge"""
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
    reasoning_file = results_dir / f"{model_name}_ICA_reasoning_scoring.json"

    if not reasoning_file.exists():
        print(f"Reasoning judge file not found: {reasoning_file}")
        return None

    print(f"Analyzing reasoning judge from {reasoning_file}")
    reasoning_data = load_json(reasoning_file)

    if not reasoning_data:
        print("Reasoning judge file is empty or invalid")
        return None

    # Initialize the statistical results
    stats = {
        "test_num": 0,
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

    # Iterate over each test item
    for item_id, item_data in reasoning_data.items():
        if not isinstance(item_data, list):
            print(
                f"Invalid data format for item {item_id}, expected list but got {type(item_data)}"
            )
            continue

        # Process each sub-test
        for sub_idx, sub_data in enumerate(item_data):
            try:
                if not isinstance(sub_data, dict):
                    print(
                        f"Invalid sub-data format for item {item_id}[{sub_idx}], expected dict but got {type(sub_data)}"
                    )
                    continue

                stats["test_num"] += 1

                # Extract hop_quality data
                hop_quality_path1 = sub_data.get("hop_quality_path1", {})
                hop_quality_path2 = sub_data.get("hop_quality_path2", {})

                # Process path1
                if hop_quality_path1:
                    # Record the hop count
                    hop_count = len(hop_quality_path1)
                    stats["hop_count_distribution"][str(hop_count)] += 1

                    # Calculate the reasoning score
                    reasoning_score = calculate_reasoning_score(hop_quality_path1)
                    total_reasoning_score += reasoning_score

                    # Record the reasoning score distribution
                    if 0 <= reasoning_score < 1:
                        stats["reasoning_score_distribution"]["0-1"] += 1
                    elif 1 <= reasoning_score < 2:
                        stats["reasoning_score_distribution"]["1-2"] += 1
                    else:
                        stats["reasoning_score_distribution"]["2-3"] += 1

                    # Calculate the average value of each dimension
                    path_reasonableness = 0
                    path_distinctiveness = 0
                    path_knowledgeability = 0

                    for hop, scores in hop_quality_path1.items():
                        if not isinstance(scores, list) or len(scores) != 3:
                            print(
                                f"Invalid hop score format for {item_id}[{sub_idx}] path1 {hop}: {scores}"
                            )
                            continue

                        path_reasonableness += scores[0]
                        path_distinctiveness += scores[1]
                        path_knowledgeability += scores[2]
                        total_hops += 1

                    if hop_count > 0:
                        path_reasonableness /= hop_count
                        path_distinctiveness /= hop_count
                        path_knowledgeability /= hop_count

                        total_reasonableness += path_reasonableness
                        total_distinctiveness += path_distinctiveness
                        total_knowledgeability += path_knowledgeability

                        # Record the distribution of each dimension
                        # Reasonableness
                        if 0 <= path_reasonableness < 0.2:
                            stats["reasonableness_distribution"]["0-0.2"] += 1
                        elif 0.2 <= path_reasonableness < 0.4:
                            stats["reasonableness_distribution"]["0.2-0.4"] += 1
                        elif 0.4 <= path_reasonableness < 0.6:
                            stats["reasonableness_distribution"]["0.4-0.6"] += 1
                        elif 0.6 <= path_reasonableness < 0.8:
                            stats["reasonableness_distribution"]["0.6-0.8"] += 1
                        else:
                            stats["reasonableness_distribution"]["0.8-1"] += 1

                        # Distinctiveness
                        if 0 <= path_distinctiveness < 0.2:
                            stats["distinctiveness_distribution"]["0-0.2"] += 1
                        elif 0.2 <= path_distinctiveness < 0.4:
                            stats["distinctiveness_distribution"]["0.2-0.4"] += 1
                        elif 0.4 <= path_distinctiveness < 0.6:
                            stats["distinctiveness_distribution"]["0.4-0.6"] += 1
                        elif 0.6 <= path_distinctiveness < 0.8:
                            stats["distinctiveness_distribution"]["0.6-0.8"] += 1
                        else:
                            stats["distinctiveness_distribution"]["0.8-1"] += 1

                        # Knowledgeability
                        if 0 <= path_knowledgeability < 0.5:
                            stats["knowledgeability_distribution"]["0-0.5"] += 1
                        else:
                            stats["knowledgeability_distribution"]["0.5-1"] += 1

                # Process path2
                if hop_quality_path2:
                    # Record the hop count
                    hop_count = len(hop_quality_path2)
                    stats["hop_count_distribution"][str(hop_count)] += 1

                    # Calculate the reasoning score
                    reasoning_score = calculate_reasoning_score(hop_quality_path2)
                    total_reasoning_score += reasoning_score

                    # Record the reasoning score distribution
                    if 0 <= reasoning_score < 1:
                        stats["reasoning_score_distribution"]["0-1"] += 1
                    elif 1 <= reasoning_score < 2:
                        stats["reasoning_score_distribution"]["1-2"] += 1
                    else:
                        stats["reasoning_score_distribution"]["2-3"] += 1

                    # Calculate the average value of each dimension
                    path_reasonableness = 0
                    path_distinctiveness = 0
                    path_knowledgeability = 0

                    for hop, scores in hop_quality_path2.items():
                        if not isinstance(scores, list) or len(scores) != 3:
                            print(
                                f"Invalid hop score format for {item_id}[{sub_idx}] path2 {hop}: {scores}"
                            )
                            continue

                        path_reasonableness += scores[0]
                        path_distinctiveness += scores[1]
                        path_knowledgeability += scores[2]
                        total_hops += 1

                    if hop_count > 0:
                        path_reasonableness /= hop_count
                        path_distinctiveness /= hop_count
                        path_knowledgeability /= hop_count

                        total_reasonableness += path_reasonableness
                        total_distinctiveness += path_distinctiveness
                        total_knowledgeability += path_knowledgeability

                        # Record the distribution of each dimension
                        # Reasonableness
                        if 0 <= path_reasonableness < 0.2:
                            stats["reasonableness_distribution"]["0-0.2"] += 1
                        elif 0.2 <= path_reasonableness < 0.4:
                            stats["reasonableness_distribution"]["0.2-0.4"] += 1
                        elif 0.4 <= path_reasonableness < 0.6:
                            stats["reasonableness_distribution"]["0.4-0.6"] += 1
                        elif 0.6 <= path_reasonableness < 0.8:
                            stats["reasonableness_distribution"]["0.6-0.8"] += 1
                        else:
                            stats["reasonableness_distribution"]["0.8-1"] += 1

                        # Distinctiveness
                        if 0 <= path_distinctiveness < 0.2:
                            stats["distinctiveness_distribution"]["0-0.2"] += 1
                        elif 0.2 <= path_distinctiveness < 0.4:
                            stats["distinctiveness_distribution"]["0.2-0.4"] += 1
                        elif 0.4 <= path_distinctiveness < 0.6:
                            stats["distinctiveness_distribution"]["0.4-0.6"] += 1
                        elif 0.6 <= path_distinctiveness < 0.8:
                            stats["distinctiveness_distribution"]["0.6-0.8"] += 1
                        else:
                            stats["distinctiveness_distribution"]["0.8-1"] += 1

                        # Knowledgeability
                        if 0 <= path_knowledgeability < 0.5:
                            stats["knowledgeability_distribution"]["0-0.5"] += 1
                        else:
                            stats["knowledgeability_distribution"]["0.5-1"] += 1

            except Exception as e:
                print(
                    f"Error processing reasoning data for item {item_id}[{sub_idx}]: {e}"
                )
                continue

    # Calculate the average value
    if stats["test_num"] > 0:
        # Considering that each test item may have path1 and path2, so divide by 2*test_num
        path_count = sum(stats["hop_count_distribution"].values())
        stats["avg_reasoning_score"] = (
            total_reasoning_score / path_count if path_count > 0 else 0
        )

        # Add average metrics
        stats["avg_reasonableness"] = (
            total_reasonableness / path_count if path_count > 0 else 0
        )
        stats["avg_distinctiveness"] = (
            total_distinctiveness / path_count if path_count > 0 else 0
        )
        stats["avg_knowledgeability"] = (
            total_knowledgeability / path_count if path_count > 0 else 0
        )

        # Calculate the distribution percentage
        for key in stats["reasoning_score_distribution"]:
            stats["reasoning_score_distribution"][key] = (
                stats["reasoning_score_distribution"][key] / path_count
                if path_count > 0
                else 0
            )

        for key in stats["hop_count_distribution"]:
            stats["hop_count_distribution"][key] = (
                stats["hop_count_distribution"][key] / path_count
                if path_count > 0
                else 0
            )

        for key in stats["reasonableness_distribution"]:
            stats["reasonableness_distribution"][key] = (
                stats["reasonableness_distribution"][key] / path_count
                if path_count > 0
                else 0
            )

        for key in stats["distinctiveness_distribution"]:
            stats["distinctiveness_distribution"][key] = (
                stats["distinctiveness_distribution"][key] / path_count
                if path_count > 0
                else 0
            )

        for key in stats["knowledgeability_distribution"]:
            stats["knowledgeability_distribution"][key] = (
                stats["knowledgeability_distribution"][key] / path_count
                if path_count > 0
                else 0
            )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze the results of ICA")
    parser.add_argument("--model_name", type=str, help="The model name to be analyzed")
    args = parser.parse_args()

    # Load the configuration
    config = get_config()

    model_name = args.model_name
    if not model_name:
        model_name = config["evaluation_settings"]["ria"]["default_model_name"]

    print(f"Start analyzing the ICA results of {model_name}...")

    # Load the dataset information
    metadata_map = load_dataset_info(config)
    metadata_map = parse_perception_conception(metadata_map)

    # Analyze the results of the regular judge
    regular_stats = analyze_regular_judge(model_name, config, metadata_map)

    # Analyze the results of the reasoning judge
    reasoning_stats = analyze_reasoning_judge(model_name, config)

    # Integrate the results
    stats = {
        "model_name": model_name,
        "regular_judge": regular_stats if regular_stats else "Not available",
        "reasoning_judge": reasoning_stats if reasoning_stats else "Not available",
    }

    # Save the results
    results_dir = PROJECT_ROOT / config["general_settings"]["results_base_dir"]
    results_file_path = results_dir / f"{model_name}_ICA_stats.json"
    save_json(stats, results_file_path)

    print(f"Statistics saved to {results_file_path}")

    # Print the summary
    print(f"Model: {model_name}")
    if regular_stats:
        print(
            f"Regular Judge - Test Count: {regular_stats['test_num']}, Avg Score: {regular_stats['score_rate']:.2f}"
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
    main()
