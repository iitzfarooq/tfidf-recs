from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class CSVLoader(BaseLoader):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.filepath: str = self.config.get("filepath", "")
        self.columns: List[str] = self.config.get("columns", None)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        if self.columns:
            df = df[self.columns]
        return df

# More loader implementations can be added here following the same pattern.