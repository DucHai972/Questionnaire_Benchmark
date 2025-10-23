# Data Processing Scripts

This folder contains utility scripts for extracting and merging benchmark data.

## Scripts

### Extraction Scripts

**`extract_empty_responses.py`**
- Extracts cases with `EMPTY_MODEL_RESPONSE` status from benchmark results
- Creates filtered directory structure for re-running benchmark on failed cases
- Useful for retry mechanisms when API calls fail

**`extract_empty_strings.py`**
- Extracts cases with empty string responses ('' vs 'None')
- Similar to extract_empty_responses but targets different empty response types
- Helps identify cases that need to be re-run

**`extract_qa_pairs.py`**
- Extracts question and expected_answer pairs from advanced_prompts JSON files
- Organizes output by dataset and task into clean CSV files
- Output location: `extracted_qa_pairs/`
- Extracts 1,500 Q&A pairs total (5 datasets × 6 tasks × 50 cases)

### Merge Scripts

**`merge_empty_to_main.py`**
- Merges results from `gpt-5-mini-empty` back into main `gpt-5-mini` results
- Filters out EMPTY_MODEL_RESPONSE entries
- Only updates actual responses, not empty placeholders

**`merge_filled_responses.py`**
- Merges re-run results from `gpt-5-mini-filled` back into main results
- Updates Response field for specific cases
- Preserves non-filled cases

## Typical Workflow

1. **Run benchmark** → Some cases return empty/failed responses
2. **Extract empty cases** → Use `extract_empty_responses.py` or `extract_empty_strings.py`
3. **Re-run benchmark** → Run benchmark on extracted cases only
4. **Merge results** → Use `merge_filled_responses.py` or `merge_empty_to_main.py`

## Usage Examples

```bash
# Extract Q&A pairs from advanced prompts
python3 extract_qa_pairs.py

# Extract empty response cases
python3 extract_empty_responses.py

# Merge filled responses back
python3 merge_filled_responses.py
```

## Output Locations

- `extract_qa_pairs.py` → `../extracted_qa_pairs/`
- `extract_empty_responses.py` → `../benchmark_results/gpt-5-mini-empty/`
- `extract_empty_strings.py` → `../benchmark_results/gpt-5-mini-empty-strings/`
