# Q_Benchmark - Language Model Structured Data Evaluation

A comprehensive benchmark system for evaluating large language model performance on structured data question-answering tasks across multiple formats (JSON, XML, HTML, Markdown, TTL, TXT) and question types.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Generate converted prompts:**
   ```bash
   # Generate base prompts
   python generate_converted_prompts.py
   
   # Generate prompt variants (optional)
   python generate_converted_prompts_variants.py
   
   # Generate self-augmentation prompts (optional)  
   python generate_converted_prompts_self_aug.py
   ```

4. **Run benchmark:**
   ```bash
   # Full benchmark
   python benchmark_pipeline.py --model openai --openai-model gpt-4o-mini
   
   # Single test
   python benchmark_pipeline.py --model openai --openai-model gpt-4o-mini --dataset healthcare-dataset --task answer_lookup --format json --max-cases 5
   ```

5. **Analyze results:**
   ```bash
   python benchmark_analysis_final.py --model gpt-4o-mini
   ```

## Datasets & Tasks

- **5 Datasets:** healthcare-dataset, isbar, self-reported-mental-health, stack-overflow-2022, sus-uta7
- **6 Task Types:** answer_lookup, answer_reverse_lookup, conceptual_aggregation, multi_hop_relational_inference, respondent_count, rule_based_querying  
- **6 Formats:** JSON, XML, HTML, Markdown, TTL, TXT

See `CLAUDE.md` for detailed documentation and usage examples.