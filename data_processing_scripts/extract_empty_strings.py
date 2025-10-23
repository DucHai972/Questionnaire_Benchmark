#!/usr/bin/env python3
"""
Extract all cases with empty string responses from gpt-5-mini

This script creates a new directory structure (gpt-5-mini-empty-strings) containing only
the cases where Response == '' (empty string, not 'None').

Similar to extract_empty_responses.py but for empty strings.
"""

import os
import csv
from pathlib import Path
import shutil

def extract_empty_string_responses(source_dir, output_dir):
    """
    Extract all cases with empty string responses from source_dir to output_dir

    Args:
        source_dir (Path): Source directory (gpt-5-mini)
        output_dir (Path): Output directory (gpt-5-mini-empty-strings)

    Returns:
        dict: Statistics about extraction
    """
    stats = {
        'files_processed': 0,
        'files_with_empty': 0,
        'total_empty_cases': 0,
        'files_created': 0
    }

    # Find all CSV files in source directory
    csv_files = list(source_dir.rglob('*.csv'))

    print(f"Found {len(csv_files)} CSV files to process")
    print()

    for csv_file in sorted(csv_files):
        stats['files_processed'] += 1

        try:
            # Read the source file
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

                if 'Response' not in fieldnames:
                    print(f"⚠️  Skipping {csv_file.name} - no Response column")
                    continue

                # Filter rows with empty string responses
                empty_rows = []
                for row in reader:
                    response = row.get('Response', '')
                    # Check for empty string (not 'None')
                    if response.strip() == '' and response != 'None':
                        empty_rows.append(row)

                # If we found empty string responses, create the output file
                if empty_rows:
                    stats['files_with_empty'] += 1
                    stats['total_empty_cases'] += len(empty_rows)

                    # Create output path with same structure
                    relative_path = csv_file.relative_to(source_dir)
                    output_file = output_dir / relative_path

                    # Create parent directories if needed
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Write filtered rows to output file
                    with open(output_file, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(empty_rows)

                    stats['files_created'] += 1
                    print(f"✓ {relative_path}")
                    print(f"  Extracted {len(empty_rows)} cases with empty string responses")

        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")

    return stats

def main():
    base_dir = Path("/insight-fast/dnguyen/Questionnaire_Benchmark/benchmark_results")
    source_dir = base_dir / "gpt-5-mini"
    output_dir = base_dir / "gpt-5-mini-empty-strings"

    print("="*80)
    print("EXTRACTING EMPTY STRING RESPONSE CASES FROM GPT-5-MINI")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    # Check if output directory exists
    if output_dir.exists():
        response = input(f"Output directory {output_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
        print(f"Removed existing directory: {output_dir}")
        print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract empty string responses
    stats = extract_empty_string_responses(source_dir, output_dir)

    # Print summary
    print()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files with empty string responses: {stats['files_with_empty']}")
    print(f"Files created: {stats['files_created']}")
    print(f"Total empty string cases extracted: {stats['total_empty_cases']}")
    print()
    print(f"✓ Extraction complete!")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("1. Run benchmark_pipeline.py on gpt-5-mini-empty-strings to fill these cases")
    print("2. Merge the filled responses back into gpt-5-mini")

if __name__ == "__main__":
    main()
