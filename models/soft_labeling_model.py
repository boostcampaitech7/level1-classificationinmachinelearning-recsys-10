from model import Model
import numpy as np
import pandas as pd
import lightgbm as lgb

class SoftLabelingModel(Model):
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        y_pct = y_price.pct_change().shift(-1).fillna(0).copy()
        y_soft_label = np.tanh(y_pct * 10)
        
        train_dataset = lgb.Dataset(X, y_soft_label)
        params = {
            'learning_rate': 0.001,
            'num_leaves': 15,
            'max_depth': -1,
            'random_state': 42,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'verbose': -1,
        }
        self.model = lgb.train(params, train_dataset)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        y_soft_label_pred = self.model.predict(X)
        soft_to_hard = lambda x: 1 if x < 0 else 2
        y_predict = pd.Series(y_soft_label_pred).apply(soft_to_hard)
        return y_predict
