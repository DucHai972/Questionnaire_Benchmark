# Q_Benchmark - Language Model Structured Data Evaluation

A comprehensive benchmark system for evaluating large language model performance on structured data question-answering tasks across multiple formats (JSON, XML, HTML, Markdown, TTL, TXT) and question types.

## Features

- **Multi-Provider Support**: OpenAI (GPT-4, GPT-3.5), Google (Gemini), AWS Bedrock (Llama, Claude, Mistral)
- **Multiple Data Formats**: JSON, XML, HTML, Markdown, TTL, TXT
- **Diverse Tasks**: Answer lookup, reverse lookup, aggregation, multi-hop inference, counting, rule-based querying
- **Prompt Variants**: Test different prompt strategies (with/without role prompting, formatting, etc.)
- **Self-Augmentation**: Enhanced prompts with structural information

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY for GPT models
# - GOOGLE_API_KEY for Gemini models
# - AWS_BEARER_TOKEN_BEDROCK for Bedrock models
# - AWS_REGION (e.g., us-east-1)
```

### 3. Generate Converted Prompts

```bash
# Generate base prompts
python generate_converted_prompts.py

# Optional: Generate prompt variants
python generate_converted_prompts_variants.py

# Optional: Generate self-augmentation prompts
python generate_converted_prompts_self_aug.py
```

### 4. Run Benchmarks

#### OpenAI (GPT Models)
```bash
# Full benchmark with GPT-4o-mini
python benchmark_pipeline.py --model openai --openai-model gpt-4o-mini

# Test specific dataset/task
python benchmark_pipeline.py \
  --model openai \
  --openai-model gpt-4o-mini \
  --dataset healthcare-dataset \
  --task answer_lookup \
  --format json \
  --max-cases 5
```

#### Google (Gemini Models)
```bash
# Full benchmark with Gemini
python benchmark_pipeline.py --model google --google-model gemini-1.5-flash

# Test specific dataset
python benchmark_pipeline.py \
  --model google \
  --google-model gemini-1.5-flash \
  --dataset healthcare-dataset \
  --max-cases 10
```

#### AWS Bedrock (Llama, Claude, Mistral)
```bash
# Llama 3.3 70B (us-east-1)
python benchmark_pipeline.py \
  --model bedrock \
  --bedrock-model us.meta.llama3-3-70b-instruct-v1:0 \
  --dataset healthcare-dataset \
  --task answer_lookup \
  --max-cases 5

# Claude 3 Haiku (check region availability)
python benchmark_pipeline.py \
  --model bedrock \
  --bedrock-model anthropic.claude-3-haiku-20240307-v1:0 \
  --dataset isbar \
  --max-cases 5

# Mistral Large (eu-west-1)
python benchmark_pipeline.py \
  --model bedrock \
  --bedrock-model mistral.mistral-large-2402-v1:0 \
  --dataset sus-uta7 \
  --max-cases 5
```

### 5. Analyze Results

```bash
# Analyze results for a specific model
python benchmark_analysis_final.py --model gpt-4o-mini

# Analyze Bedrock model results
python benchmark_analysis_final.py --model us.meta.llama3-3-70b-instruct-v1:0
```

## Available Options

### Datasets (5)
- `healthcare-dataset`
- `isbar`
- `self-reported-mental-health`
- `stack-overflow-2022`
- `sus-uta7`

### Task Types (6)
- `answer_lookup` - Find specific values
- `answer_reverse_lookup` - Find respondents matching criteria
- `conceptual_aggregation` - Aggregate/count values
- `multi_hop_relational_inference` - Multi-step reasoning
- `respondent_count` - Count respondents matching conditions
- `rule_based_querying` - Apply rules to find answers

### Data Formats (6)
JSON, XML, HTML, Markdown, TTL, TXT

### Command-Line Options

```bash
--dataset DATASET          # Specific dataset to process
--task TASK               # Specific task type
--format FORMAT           # Data format (json, xml, html, md, ttl, txt)
--model {openai,google,bedrock}  # Model provider
--openai-model MODEL      # OpenAI model name (default: gpt-3.5-turbo)
--google-model MODEL      # Google model name (default: gemini-1.5-flash)
--bedrock-model MODEL     # Bedrock model ID
--max-cases N             # Limit number of cases per file
--start-case N            # Start from specific case number (default: 2)
--variants TYPE           # Use prompt variants
--self_aug TYPE           # Use self-augmentation prompts
--list                    # List available options
```

## Supported Models

### AWS Bedrock Models
**Note:** Model availability depends on your AWS region and permissions.

**US Regions (us-east-1, us-west-2):**
- `us.meta.llama3-3-70b-instruct-v1:0` - Llama 3.3 70B
- `us.anthropic.claude-3-5-sonnet-20240620-v1:0` - Claude 3.5 Sonnet
- `us.anthropic.claude-3-opus-20240229-v1:0` - Claude 3 Opus

**EU Regions (eu-west-1):**
- `mistral.mistral-large-2402-v1:0` - Mistral Large
- `mistral.mistral-7b-instruct-v0:2` - Mistral 7B
- `amazon.titan-text-express-v1` - Amazon Titan
- `anthropic.claude-3-haiku-20240307-v1:0` - Claude 3 Haiku

**Check model availability in your region** using the AWS Console or contact your administrator.

## Project Structure

```
Questionnaire_Benchmark/
├── benchmark_pipeline.py                    # Main benchmark runner
├── benchmark_analysis_final.py              # Results analyzer
├── bedrock_client.py                        # AWS Bedrock API client
├── generate_converted_prompts.py            # Base prompt generator
├── generate_converted_prompts_variants.py   # Variant prompt generator
├── generate_converted_prompts_self_aug.py   # Self-augmentation prompt generator
├── improved_evaluation.py                   # Enhanced evaluation functions
├── robust_csv_parser_improved.py            # Robust CSV parser with improved evaluation
├── data_processing_scripts/                 # Utility scripts for data processing
├── advanced_prompts/                        # Source questionnaire data
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment template
└── README.md                                # This file
```

### Generated Directories (Not in Git)
- `converted_prompts/` - Generated prompts
- `benchmark_results/` - Test results
- `analysis_results_final/` - Analysis outputs

## Environment Variables

Create a `.env` file with:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=AIza...

# AWS Bedrock
AWS_BEARER_TOKEN_BEDROCK=your-bearer-token
AWS_REGION=us-east-1
```

## Tips

1. **Start Small**: Use `--max-cases 5` for initial tests
2. **Check Region**: Verify model availability in your AWS region
3. **Monitor Costs**: LLM API calls can be expensive for large benchmarks
4. **Parallel Processing**: Run different datasets in parallel for faster results
5. **Review Logs**: Check console output for errors or warnings

## Troubleshooting

### AWS Bedrock Issues

**403 Forbidden Error:**
- Your Bearer token may not have access to that region
- Check `AWS_REGION` in `.env` matches your token's permissions

**404 Model Not Found:**
- Model may not be available in your region
- Try a different model or contact AWS support

**Empty Responses:**
- Check API credentials are valid
- Verify model has access permissions
- Review CloudWatch logs (AWS)

### Common Issues

**"No prompts found":**
- Run `generate_converted_prompts.py` first

**"Client initialization failed":**
- Check API keys in `.env` file
- Verify API key format is correct