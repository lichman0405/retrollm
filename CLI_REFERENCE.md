# RetroLLM CLI Reference

This document describes all command-line interfaces exposed by `retrollm`.

## Prerequisites

Before running `smoke` or `search`:

1. Activate your environment (Conda or venv).
2. Install the package:
   ```bash
   python -m pip install -e .
   ```
3. Download artifacts:
   ```bash
   retrollm download-data ./data
   ```

You can validate your setup with:

```bash
retrollm doctor --config ./data/config.yml
```

## Global Usage

```bash
retrollm [-h] {download-data,smoke,search,doctor} ...
```

Global option:

1. `-h, --help`: Show help for the top-level CLI.

## Command: `download-data`

Downloads public artifacts and writes `config.yml`.

Usage:

```bash
retrollm download-data <path>
```

Arguments:

1. `path` (required, positional): Target directory for downloaded artifacts.

Expected output:

1. ONNX models, template files, stock file in `<path>`.
2. A generated config at `<path>/config.yml`.

## Command: `smoke`

Loads the policy ONNX model and runs one forward pass for a target molecule.

Usage:

```bash
retrollm smoke --config <path> --smiles <SMILES> [--topk <int>]
```

Arguments:

1. `--config` (required): Path to artifact config (`config.yml`).
2. `--smiles` (required): Target molecule SMILES.
3. `--topk` (optional, default: `50`): Number of top predictions to return.

Expected output:

1. Top-k template indices with probabilities.

## Command: `search`

Runs standalone MCTS retrosynthesis search, with optional LLM-assisted workflow.

Usage:

```bash
retrollm search --config <path> --smiles <SMILES> [options]
```

### Required Arguments

1. `--config`: Path to `config.yml`.
2. `--smiles`: Target molecule SMILES.

### Core Search Arguments

1. `--iteration-limit` (int, default: `150`): Maximum MCTS iterations.
2. `--time-limit` (float, default: `120.0`): Search time budget in seconds.
3. `--c-puct` (float, default: `1.4`): MCTS exploration constant.
4. `--topk` (int, default: `20`): Top-k template proposals per expansion step.
5. `--max-depth` (int, default: `6`): Maximum tree depth.
6. `--max-routes` (int, default: `3`): Number of output routes.

### Expansion/Policy Control

1. `--no-filter` (flag): Disable ONNX filter policy.
2. `--filter-cutoff` (float, default: `0.0`): Minimum filter score threshold.
3. `--no-ringbreaker` (flag): Disable ringbreaker fallback policy.
4. `--ringbreaker-topk` (int, default: `10`): Top-k for ringbreaker expansion.

### LLM Workflow Control

1. `--use-llm` (flag): Enable LLM workflow.
2. `--llm-interval` (int, default: `50`): Iteration interval for meta-control updates.
3. `--constraints` (string, default: empty): Natural-language search constraints.
4. `--no-llm-advisor` (flag): Disable LLM subgoal advisor.
5. `--no-llm-rerank` (flag): Disable LLM route reranking.
6. `--no-llm-diagnosis` (flag): Disable LLM failure diagnosis.
7. `--no-llm-handoff` (flag): Disable LLM handoff generation.
8. `--no-llm-retry` (flag): Disable retry after failure.
9. `--llm-max-retries` (int, default: `1`): Maximum retry attempts after diagnosis.

### Output and Reporting

1. `--json` (flag): Print raw JSON output to terminal.
2. `--output <path>`: Save full JSON output to file.
3. `--report <path>`: Save handoff markdown report to file.
4. `--verbose` (flag): Print additional diagnostics (including LLM events).

### Search Output Sections (default human-readable mode)

1. Search Summary
2. Constraint Translation (if available)
3. Route Rerank (if available)
4. Failure Diagnosis (if available)
5. Route tables
6. Handoff Preview (if available)
7. LLM Events (with `--verbose`)

### Notes

1. Without `--use-llm`, core search still works (policy/filter/ringbreaker).
2. `--constraints` still has heuristic handling even when LLM is disabled.
3. `--report` writes only if handoff text is available.

## Command: `doctor`

Checks environment, dependencies, LLM config, and artifacts.

Usage:

```bash
retrollm doctor [--config <path>] [--json] [--output <path>]
```

Arguments:

1. `--config` (optional, default: `./data/config.yml`): Artifact config path.
2. `--json` (flag): Print JSON report instead of table.
3. `--output <path>`: Save JSON report to file.

Checks include:

1. Python version compatibility (`>=3.10,<3.13`)
2. Virtual environment status
3. Package dependencies (NumPy, Pandas, RDKit, ONNX Runtime, etc.)
4. LLM configuration completeness
5. Artifact file existence and size

Exit code:

1. `0`: No failed checks.
2. `1`: At least one failed check.

## LLM Environment Variables

The CLI loads `.env` from project root (if present).

Common variables:

1. `RETROLLM_LLM_PROVIDER`
2. `RETROLLM_LLM_BASE_URL`
3. `RETROLLM_LLM_API_KEY`
4. `RETROLLM_LLM_MODEL`
5. `RETROLLM_LLM_TEMPERATURE` (optional)
6. `RETROLLM_LLM_TIMEOUT_S` (optional)

Supported OpenAI-compatible aliases include:

1. `openai_compatible`
2. `deepseek`
3. `openrouter`
4. `siliconflow`
5. `qwen`
6. `kimi`

You can also use a custom provider class path:

1. `package.module:ClassName`

## Practical Examples

### Minimal Search

```bash
retrollm search \
  --config ./data/config.yml \
  --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

### LLM-Enabled Search with Constraints

```bash
retrollm search \
  --config ./data/config.yml \
  --smiles "CCCCCCCCCCCCCCCC(=O)NCCNCCN" \
  --use-llm \
  --constraints "avoid palladium catalysts and keep route within 6 steps" \
  --report ./handoff.md \
  --verbose
```

### JSON Output for Automation

```bash
retrollm search \
  --config ./data/config.yml \
  --smiles "CCCCCCCCCCCCCCCC(=O)NCCNCCN" \
  --json \
  --output ./result.json
```
