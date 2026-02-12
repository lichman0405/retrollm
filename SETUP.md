# Setup Guide

This project depends on native scientific packages (notably RDKit), so environment setup matters.

## Recommendation (Most Reliable): Conda/Mamba

If you want the fewest dependency issues across Windows/macOS/Linux, use Conda-compatible tooling (`conda`, `mamba`, or `micromamba`).

### 1) Create and activate environment

```bash
conda env create -f environment.yml
conda activate retrollm
```

### 2) Install project in editable mode

```bash
python -m pip install -e .
```

### 3) Download model/data artifacts

```bash
retrollm download-data ./data
```

If `retrollm` is not on `PATH`, use:

```bash
python -m retrollm.cli download-data ./data
```

### 4) Verify setup

```bash
retrollm doctor --config ./data/config.yml
```

or:

```bash
python -m retrollm.cli doctor --config ./data/config.yml
```

### 5) Smoke test

```bash
retrollm smoke --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

## Optional: venv + pip

You can try `venv`, but it is less reliable for RDKit across platforms/Python versions.
If install fails, switch back to Conda (recommended).

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### macOS/Linux (bash/zsh)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then run:

```bash
python -m retrollm.cli doctor --config ./data/config.yml
```

## Daily Workflow (Any Platform)

Each new terminal session:

1. Activate your environment (`conda activate retrollm` or activate `.venv`).
2. Run commands/tests.

Typical commands:

```bash
pytest -q
retrollm doctor --config ./data/config.yml
retrollm search --config ./data/config.yml --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

## Notes

- Deleting local env directories (for example `.conda` or `.venv`) is safe for the repository itself.
- The project requires Python `>=3.10,<3.13` (Python 3.11 is the safest choice).
