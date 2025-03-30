from model import Model
import numpy as np
import pandas as pd
import lightgbm as lgb

class SoftLabelingModel(Model):
    """soft labeling 모델을 사용하여 주가 등락을 예측하는 모델입니다.

    LightGBM을 사용하여 데이터의 soft label를 예측하고, 이를 클래스로 변환 후 반환합니다.
    soft label은 tanh 함수를 통해 변환된 price의 변화량입니다.
    구체적으로, label = tanh(price_pct_change * smoothness)입니다.
    또한 모델은 가격 변화의 방향만 집중하기에, 오직 label 1과 2만을 예측 결과로 내보냅니다.
    """
    def __init__(self, model_params: dict = None, selected_features: list = None, smoothness: int = 10):
        """soft labeling 모델의 각종 설정을 초기화합니다.

        Parameters
        ----------
        model_params : dict, optional
            lightgbm.train에 들어가는 파라미터들을 정의한 딕셔너리입니다.
            전달하지 않은 경우 lightgbm의 default hyperparameter를 사용합니다.
        selected_features : list, optional
            사용할 feature들의 이름을 담은 리스트입니다.
            전달하지 않은 경우 모든 feature를 사용합니다.
        smoothness : int, optional
            soft labeling의 smoothness를 결정하는 파라미터입니다.
            smoothness가 높을수록 모델이 price의 방향에 더 민감하게 반응하며 분류모델에 가까워집니다.
            smoothness가 낮을수록 모델이 price의 변화량에 대한 회귀모델에 가까워집니다.
            Default는 10입니다.
        """
        if model_params is None:
            model_params = {
                'random_state': 42,
                'verbose': -1,
            }
        if selected_features is None:
            selected_features = 'all'

        self.model_params = model_params
        self.smoothness = smoothness
        self.selected_features = selected_features

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]
        y_pct = (y_price.pct_change() * 100).shift(-1).fillna(0).copy()
        y_soft_label = np.tanh(y_pct * self.smoothness)
        train_dataset = lgb.Dataset(selected_X, y_soft_label)
        self.model = lgb.train(self.model_params, train_dataset)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]
        y_soft_label_pred = self.model.predict(selected_X)
        soft_to_hard = lambda x: 1 if x < 0 else 2
        y_predict = pd.Series(y_soft_label_pred).apply(soft_to_hard)
        return y_predict
