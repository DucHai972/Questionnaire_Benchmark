#!/usr/bin/env python3
"""
Merge filled responses from gpt-5-mini-empty back into gpt-5-mini

This script updates the Response field in gpt-5-mini for cases that were
filled in gpt-5-mini-empty, without affecting any other cases.
"""

import os
import csv
from pathlib import Path
import shutil

def merge_responses(original_file, filled_file, output_file):
    """
    Merge responses from filled_file into original_file
    Returns: (total_cases, updated_cases)
    """
    # Read filled responses into a dictionary
    filled_responses = {}
    try:
        with open(filled_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            response_col = 'Response' if 'Response' in reader.fieldnames else 'response'

            for row in reader:
                case_id = row.get('case_id', '')
                if case_id:
                    filled_responses[case_id] = row.get(response_col, '')
    except Exception as e:
        print(f"  Error reading filled file: {e}")
        return 0, 0

    # Read original file and update responses
    updated_rows = []
    total_cases = 0
    updated_cases = 0

    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            response_col = 'Response' if 'Response' in fieldnames else 'response'

            for row in reader:
                total_cases += 1
                case_id = row.get('case_id', '')

                # If this case has a filled response, update it
                # Only update if the filled response is not EMPTY_MODEL_RESPONSE
                if case_id in filled_responses:
                    new_response = filled_responses[case_id]
                    if new_response != 'EMPTY_MODEL_RESPONSE':
                        old_response = row[response_col]
                        row[response_col] = new_response
                        updated_cases += 1
                        print(f"    Updated {case_id}: '{old_response[:50]}...' -> '{new_response[:50]}...'")

                updated_rows.append(row)

        # Write updated data to output file
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        return total_cases, updated_cases

    except Exception as e:
        print(f"  Error processing original file: {e}")
        return 0, 0

def main():
    base_dir = Path("/insight-fast/dnguyen/Questionnaire_Benchmark/benchmark_results")
    gpt5mini_dir = base_dir / "gpt-5-mini"
    empty_dir = base_dir / "gpt-5-mini-empty"

    # Find all CSV files in gpt-5-mini-empty
    filled_files = list(empty_dir.rglob("*.csv"))
    print(f"Found {len(filled_files)} filled CSV files to merge\n")

    total_files_processed = 0
    total_cases_updated = 0
    errors = []

    for filled_file in filled_files:
        # Construct the corresponding path in gpt-5-mini
        relative_path = filled_file.relative_to(empty_dir)
        original_file = gpt5mini_dir / relative_path

        if not original_file.exists():
            errors.append(f"Original file not found: {relative_path}")
            continue

        print(f"Processing: {relative_path}")

        # Create a temporary file for the merge
        temp_file = original_file.with_suffix('.tmp')

        # Merge responses
        total, updated = merge_responses(original_file, filled_file, temp_file)

        if total > 0:
            # Replace original with merged file
            shutil.move(str(temp_file), str(original_file))
            print(f"  ✓ Updated {updated}/{total} cases")
            total_files_processed += 1
            total_cases_updated += updated
        else:
            # Remove temp file if merge failed
            if temp_file.exists():
                temp_file.unlink()
            errors.append(f"Failed to merge: {relative_path}")

        print()

    print("\n" + "="*80)
    print("MERGE SUMMARY")
    print("="*80)
    print(f"Files processed: {total_files_processed}/{len(filled_files)}")
    print(f"Total cases updated: {total_cases_updated}")

    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✓ All files merged successfully!")

if __name__ == "__main__":
    main()
