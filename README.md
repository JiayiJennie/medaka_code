# Medaka

An LLM-based planning approach for LiquidWorld problems, from the paper *MEDAKA: Means-Ends Decomposition for Continuous-State LLM Planning*.

## Project Structure

```
medaka_code/
├── .env                       # API keys (create this, see Setup)
├── liquidword/                # Problem domain & generator
│   ├── domain/
│   │   └── liquid_world.py    # LiquidWorld state & action definitions
│   ├── data/                  # Benchmark datasets (level1–4)
│   ├── liquid_world_generator.py  # main entry for liquidworld problem generator
│   ├── config.json            # Generator config example
│   └── README.md              # Generator documentation
├── src/                       # Planning & evaluation pipeline
│   ├── main.py                # Entry point
│   ├── medaka/
│   │   └── core.py            # Main medaka class 
│   ├── models/
│   │   └── llm_client.py      # LLM API client (multi-provider)
│   ├── eval/
│   │   └── plan_val.py        # Plan validation
│   └── utils/
│       ├── config.py          # Env-based configuration
│       └── plan_parser.py     # Plan text parsing
└── README.md
```

## Setup

### 1. Install dependencies

```bash
conda create -n medaka python=3.10 -y
conda activate medaka
pip install openai anthropic google-genai python-dotenv scipy
```

### 2. Configure API keys

Create a `.env` file in the project root with your API credentials. Only the provider you plan to use needs to be filled in. Example:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o
```

See `src/utils/config.py` for the full list of supported environment variables.

## Usage

Run from the project root (`medaka_code/`):

```bash
# Solve a single problem
python3 -m src.main --provider openai  --problem liquidword/data/level2.json --id 1  

# Solve all problems in a dataset
python3 -m src.main --provider openai --problem liquidword/data/level2.json --all --num-trials 3
```

### Key options


| Flag             | Description                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `--provider`     | LLM provider (`openai`, `anthropic`, `google`, `azure_openai`, `dashscope`, 
`openrouter`, `together`, `featherless`) |
| `--model`        | Override the model name configured in `.env`                                                                         |
| `--problem`      | Path to problem JSON file                                                                                            |
| `--id`           | Problem ID(s) to solve (repeatable)                                                                                  |
| `--all`          | Solve all problems in the file                                                                                       |
| `--model`        | Override the model name configured in `.env`                                                                         |
| `--num-trials`   | Number of independent trials per problem (default: 1)                                                                |
| `--concurrency`  | Parallel trials (default: 1)                                                                                         |
| `--temperature`  | Sampling temperature (default: 0.0)                                                                                  |
| `--output`       | Custom output path (default: `results/run_<timestamp>.json`)                                                         |
| `--print-prompt` | Print prompt and exit (no API call)                                                                                  |


## Generate LiquidWorld Instences

See `[liquidword/README.md](liquidword/README.md)` for details on generating new LiquidWorld problems.

```bash
python3 -m liquidword.liquid_world_generator --config liquidword/config.json --output liquidword/data/custom.json
```

