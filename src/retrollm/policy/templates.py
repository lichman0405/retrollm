"""Template library loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class TemplateLibrary:
    """Loads reaction templates from CSV/CSV.GZ and provides index access."""

    def __init__(self, template_file: str | Path):
        self.template_file = Path(template_file)
        self.df = self._load_dataframe()
        self._smarts_col = self._detect_smarts_column()

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.template_file)
        if len(df.columns) == 1 and "\t" in str(df.columns[0]):
            df = pd.read_csv(self.template_file, sep="\t")
        return df

    def _detect_smarts_column(self) -> str:
        for candidate in (
            "retro_template",
            "template",
            "smarts",
            "reaction_smarts",
        ):
            if candidate in self.df.columns:
                return candidate
        raise ValueError(
            f"Template file {self.template_file} has no recognized SMARTS column"
        )

    def __len__(self) -> int:
        return len(self.df)

    def smarts_by_index(self, index: int) -> str:
        if index < 0 or index >= len(self.df):
            raise IndexError(index)
        return str(self.df.iloc[index][self._smarts_col])
