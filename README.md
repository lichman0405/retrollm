<div align="center">

# 🧪 RetroLLM

**Standalone neural-guided retrosynthesis search with pluggable LLM meta-control**

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square)](https://github.com/lichman0405/retrollm)
[![GitHub stars](https://img.shields.io/github/stars/lichman0405/retrollm?style=flat-square&logo=github)](https://github.com/lichman0405/retrollm/stargazers)
[![Conda](https://img.shields.io/badge/Conda-Forge-44A833?style=flat-square&logo=anaconda&logoColor=white)](environment.yml)

[![RDKit](https://img.shields.io/badge/RDKit-2023.9+-ee4c2c?style=flat-square&logo=molecule&logoColor=white)](https://www.rdkit.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-gray?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![NumPy](https://img.shields.io/badge/NumPy-<2-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+/2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Rich](https://img.shields.io/badge/Rich-13+-000?style=flat-square&logo=terminal&logoColor=white)](https://github.com/Textualize/rich)
[![PyYAML](https://img.shields.io/badge/PyYAML-6+-yellow?style=flat-square)](https://pyyaml.org/)
[![pytest](https://img.shields.io/badge/pytest-7+-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

</div>

---

RetroLLM is a standalone, minimal reproduction of the core AiZynthFinder pipeline.
It does **not** import or depend on `aizynthfinder` code, but it can reuse the **public data artifacts** (ONNX models, template library, and stock file) that AiZynthFinder publishes.

📖 **Docs** &nbsp;→&nbsp; [`SETUP.md`](SETUP.md) · [`CLI_REFERENCE.md`](CLI_REFERENCE.md)

---

## ⚡ Quick Start (Conda recommended)

```bash
git clone https://github.com/lichman0405/retrollm.git
cd retrollm
```

RDKit is best installed via Conda.

```bash
conda env create -f environment.yml
conda activate retrollm
python -m pip install -e .
```

**Download the public artifacts** (same URLs as AiZynthFinder):

```bash
retrollm download-data ./data
```

**Check environment and artifacts:**

```bash
retrollm doctor --config ./data/config.yml
```

**Run a smoke test** (loads ONNX and performs one forward pass):

```bash
retrollm smoke --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

**Run search:**

```bash
retrollm search --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

Search output is pretty-printed by default for terminal readability.
Use `--json` for machine-readable output, and `--output result.json` to save JSON.

### 🔬 Search Policies

`retrollm search` uses:

| Policy | Flag to disable |
|---|---|
| USPTO expansion policy | — |
| USPTO filter policy | `--no-filter` |
| Ringbreaker fallback expansion | `--no-ringbreaker` |

---

## 🧾 Understanding the JSON Output

When you run `retrollm search` with `--json` or `--output result.json`, each route contains a list of `steps`.
Some fields are model- or template-library specific, so the raw JSON can look confusing at first.

### Template fields

- `template_index`
  - An integer ID predicted by the expansion policy.
  - It is used as a row index into the template library CSV (see `expansion.uspto[1]` in `data/config.yml`).
- `template_smarts`
  - The reaction SMARTS string for that template.
  - RetroLLM uses templates in retrosynthesis direction:
    - `product_pattern >> reactant1.reactant2...`
  - Atom-mapping tags like `:12` improve machine alignment but reduce readability.

To pretty-print a template SMARTS in a human-friendly way:

```python
from retrollm.utils.smarts import format_reaction_smarts

print(format_reaction_smarts(template_smarts, simplify=True))
```

If you run `retrollm search --verbose`, the CLI also prints a readable multi-line view of each step's `template_smarts`.

### Model scores

- `policy_probability`
  - The expansion policy model's output score for the chosen template.
  - It is also used as the MCTS prior when selecting which child node to explore.
- `filter_score`
  - The filter model score for the proposed (product, reactants) reaction candidate.
  - If `filter_score < filter_cutoff`, the candidate is discarded.

### Route `score`

Each route's `score` is the MCTS node value estimate:

- `score = q_value = value_sum / visits`

Each simulation evaluates a node with a simple heuristic:

- Let $r$ be the fraction of molecules in the current state that are found in stock.
- If all molecules are in stock, value is `1.0`.
- Otherwise:
  - `value = max(0, r - depth_penalty)`
  - `depth_penalty = min(depth/max_depth, 1) * 0.2`

So higher `score` generally means "more of the current molecules are already purchasable" while avoiding overly deep routes.

### 🤖 Full LLM Workflow

Enable meta-control, constraint translation, subgoal advisor, route reranking, failure diagnosis/retry, and handoff draft:

```bash
retrollm search \
  --config ./data/config.yml \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --use-llm \
  --constraints "avoid Pd catalysts and keep route within 6 steps" \
  --report ./handoff.md \
  --verbose
```

---

## 🔧 LLM Configuration

Create and fill `.env` in the project root (already present, gitignored).

The LLM system is **provider-pluggable**:

| Provider type | Value |
|---|---|
| Built-in | `openai_compatible` |
| Aliases | `deepseek` · `openrouter` · `siliconflow` · `qwen` · `kimi` |
| Custom | `yourpkg.yourmod:YourProvider` |

**Required environment variables** (OpenAI-compatible provider):

| Variable | Description |
|---|---|
| `RETROLLM_LLM_BASE_URL` | API base URL |
| `RETROLLM_LLM_API_KEY` | API key |
| `RETROLLM_LLM_MODEL` | Model name |

<details>
<summary>📋 Example <code>.env</code> (DeepSeek endpoint)</summary>

```env
RETROLLM_LLM_PROVIDER=deepseek
RETROLLM_LLM_BASE_URL=https://api.deepseek.com
RETROLLM_LLM_API_KEY=your_key
RETROLLM_LLM_MODEL=deepseek-chat
```

</details>

---

## 🧑‍💻 Development

```bash
pytest -q
```

---

## 🙏 Acknowledgements

This project is inspired by and built upon the ideas from **[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)** — a tool for retrosynthetic planning powered by neural networks, developed by the Molecular AI group at AstraZeneca.

RetroLLM reuses the **public data artifacts** (ONNX models, template library, and stock file) that AiZynthFinder publishes, while providing a standalone, minimal reimplementation with pluggable LLM support.

> ⭐ If you find this project useful, please also give a **star** to the original
> **[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)** repo — huge thanks
> to the AiZynthFinder developers and the Molecular AI team for their outstanding
> open-source contribution to the computational chemistry community!

---

<div align="center">
<sub>Built with ❤️ for computational chemistry</sub>
</div>
