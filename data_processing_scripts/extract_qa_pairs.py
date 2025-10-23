#!/usr/bin/env python3
"""
Extract Question-Answer Pairs from Advanced Prompts

This script extracts only the question and expected_answer fields from
all advanced_prompts JSON files and saves them in a clean format.
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict

def extract_qa_pairs():
    """Extract Q&A pairs from all advanced_prompts JSON files."""

    base_dir = Path("/insight-fast/dnguyen/Questionnaire_Benchmark/advanced_prompts")
    output_dir = Path("/insight-fast/dnguyen/Questionnaire_Benchmark/extracted_qa_pairs")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all JSON files (excluding backups)
    json_files = sorted([f for f in base_dir.glob("**/*.json") if "backup" not in str(f)])

    print(f"Found {len(json_files)} JSON files")
    print(f"Output directory: {output_dir}\n")

    # Statistics
    stats = defaultdict(lambda: defaultdict(int))

    # Process each JSON file
    for json_file in json_files:
        # Extract dataset and task from filename
        # Example: healthcare-dataset_answer_lookup_qa_pairs.json
        filename = json_file.stem  # Remove .json extension
        parts = filename.rsplit('_qa_pairs', 1)[0]  # Remove _qa_pairs suffix

        # Split dataset and task
        # Find the last occurrence of task patterns
        tasks = ['answer_lookup', 'answer_reverse_lookup', 'conceptual_aggregation',
                 'multi_hop_relational_inference', 'respondent_count', 'rule_based_querying']

        task = None
        dataset = None
        for t in tasks:
            if parts.endswith('_' + t):
                task = t
                dataset = parts[:-len(t)-1]  # Remove task and underscore
                break

        if not task or not dataset:
            print(f"Warning: Could not parse filename: {json_file.name}")
            continue

        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract Q&A pairs
        qa_pairs = []
        for item in data:
            qa_pairs.append({
                'case_id': item.get('case_id', ''),
                'task': item.get('task', task),
                'question': item.get('question', ''),
                'expected_answer': item.get('expected_answer', '')
            })

        # Update statistics
        stats[dataset][task] = len(qa_pairs)

        # Create dataset output directory
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)

        # Save to CSV
        csv_filename = dataset_dir / f"{task}_qa_pairs.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['case_id', 'task', 'question', 'expected_answer'])
            writer.writeheader()
            writer.writerows(qa_pairs)

        print(f"✓ {dataset}/{task}: {len(qa_pairs)} Q&A pairs → {csv_filename.relative_to(output_dir)}")

    # Print summary statistics
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)

    for dataset in sorted(stats.keys()):
        print(f"\n{dataset}:")
        total = 0
        for task in sorted(stats[dataset].keys()):
            count = stats[dataset][task]
            total += count
            print(f"  {task:40s}: {count:4d} Q&A pairs")
        print(f"  {'TOTAL':40s}: {total:4d} Q&A pairs")

    # Calculate grand total
    grand_total = sum(sum(tasks.values()) for tasks in stats.values())
    print(f"\n{'GRAND TOTAL':40s}: {grand_total:4d} Q&A pairs")

    # Save summary to file
    summary_file = output_dir / "extraction_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Q&A PAIRS EXTRACTION SUMMARY\n")
        f.write("="*80 + "\n\n")

        for dataset in sorted(stats.keys()):
            f.write(f"\n{dataset}:\n")
            total = 0
            for task in sorted(stats[dataset].keys()):
                count = stats[dataset][task]
                total += count
                f.write(f"  {task:40s}: {count:4d} Q&A pairs\n")
            f.write(f"  {'TOTAL':40s}: {total:4d} Q&A pairs\n")

        f.write(f"\n{'GRAND TOTAL':40s}: {grand_total:4d} Q&A pairs\n")

    print(f"\nSummary saved to: {summary_file}")
    print(f"\nAll Q&A pairs extracted to: {output_dir}")


if __name__ == '__main__':
    extract_qa_pairs()
