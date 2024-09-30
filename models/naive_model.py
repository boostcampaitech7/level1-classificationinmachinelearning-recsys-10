from model import Model
import numpy as np
import pandas as pd

class NaiveModel(Model):

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(np.ones(len(X)))