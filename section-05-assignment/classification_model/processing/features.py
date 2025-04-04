from typing import List

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    """Extract the first letter of a variable"""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit (self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


class GetTitleTransformer(BaseEstimator, TransformerMixin):
    """Extract the title (Mr, Ms, etc) from the name variable"""

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    @staticmethod
    def _get_title(passenger):
        line = passenger
        res = re.search('Mrs|Mr|Miss|Master', line)
        if res:
            return res.group()
        return 'Other'

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['title'] = X['name'].apply(self._get_title)
        return X


class GetFirstCabinTransformer(BaseEstimator, TransformerMixin):
    """Extract the first letter of the cabin variable"""

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    @staticmethod
    def _get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['cabin'] = X['cabin'].apply(self._get_first_cabin)
        return X


class CastNumericalTransformer(BaseEstimator, TransformerMixin):
    """Convert numerical values into floats"""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].astype(float)
        return X