import pandas as pd


class SourceBase:
    name = "Base"

    def merge(self, df: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError("Plugins must implement merge()")
