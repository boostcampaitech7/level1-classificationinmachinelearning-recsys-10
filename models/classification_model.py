from model import Model
import numpy as np
import pandas as pd
import lightgbm as lgb

class ClassificationModel(Model):
    """단순 분류 모델을 사용하여 주가 등락을 예측하는 모델입니다.
    """

    def __init__(self, model_params: dict = None, selected_features: list = None, ignore_strength: bool = False):
        """분류 모델의 각종 설정을 초기화합니다.

        Parameters
        ----------
        model_params : dict, optional
            lightgbm.train에 들어가는 파라미터들을 정의한 딕셔너리입니다.
            전달하지 않은 경우 lightgbm의 default hyperparameter를 사용합니다.
        selected_features : list, optional
            사용할 feature들의 이름을 담은 리스트입니다.
            전달하지 않은 경우 모든 feature를 사용합니다.
        ignore_strength : bool, optional
            True인 경우, 예측 label 중 0, 3은 1, 2로 변경합니다.
            Default는 False입니다.
        """
        if model_params is None:
            model_params = {
                'random_state': 42,
                'verbose': -1,
            }
        if selected_features is None:
            selected_features = 'all'

        self.model_params = model_params
        self.selected_features = selected_features
        self.ignore_strength = ignore_strength

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]
        if self.ignore_strength:
            y = y.apply(lambda x: 1 if x <= 1 else 2)
        train_dataset = lgb.Dataset(selected_X, y)
        self.model = lgb.train(self.model_params, train_dataset)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]
        y_predict = self.model.predict(selected_X)
        y_predict = pd.Series(y_predict).astype(int)
        return y_predict
