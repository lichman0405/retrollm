# RetroLLM

RetroLLM is a standalone, minimal reproduction of the core AiZynthFinder pipeline.
It does **not** import or depend on `aizynthfinder` code, but it can reuse the **public data artifacts** (ONNX models, template library, and stock file) that AiZynthFinder publishes.

This repository is intentionally English-only for code, docs, and comments.

For full cross-platform environment instructions, see `SETUP.md`.

## Quick start (Conda recommended)

RDKit is best installed via Conda.

```bash
conda env create -f environment.yml
conda activate retrollm
python -m pip install -e .
```

Download the public artifacts (same URLs as AiZynthFinder):

```bash
retrollm download-data ./data
```

Check environment and artifacts:

```bash
retrollm doctor --config ./data/config.yml
```

Run a smoke test (loads ONNX and performs one forward pass):

```bash
retrollm smoke --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

Run search:

```bash
retrollm search --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

Search output is pretty-printed by default for terminal readability.
Use `--json` for machine-readable output, and `--output result.json` to save JSON.

`retrollm search` now uses:

- USPTO expansion policy
- USPTO filter policy (`--no-filter` to disable)
- Ringbreaker fallback expansion (`--no-ringbreaker` to disable)

Enable full LLM workflow (meta-control + constraint translation + subgoal advisor +
route reranking + failure diagnosis/retry + handoff draft):

```bash
retrollm search \
  --config ./data/config.yml \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --use-llm \
  --constraints "avoid Pd catalysts and keep route within 6 steps" \
  --report ./handoff.md \
  --verbose
```

## LLM configuration

Create and fill `.env` in the project root (already present, gitignored).

The LLM system is provider-pluggable.

- Built-in alias: `RETROLLM_LLM_PROVIDER=openai_compatible`
- Common OpenAI-compatible aliases also supported: `deepseek`, `openrouter`, `siliconflow`, `qwen`, `kimi`
- Custom provider: `RETROLLM_LLM_PROVIDER=yourpkg.yourmod:YourProvider`

The OpenAI-compatible provider uses:

- `RETROLLM_LLM_BASE_URL`
- `RETROLLM_LLM_API_KEY`
- `RETROLLM_LLM_MODEL`

Example (`DeepSeek` endpoint, OpenAI-compatible API):

```env
RETROLLM_LLM_PROVIDER=deepseek
RETROLLM_LLM_BASE_URL=https://api.deepseek.com
RETROLLM_LLM_API_KEY=your_key
RETROLLM_LLM_MODEL=deepseek-chat
```

## Development

```bash
pytest -q
```
