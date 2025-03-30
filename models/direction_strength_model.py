import numpy as np
import pandas as pd
from model import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class DirectionStrengthModel(Model):
    """
    XGBoost와 RandomForestRegressor를 결합한 앙상블 모델.
    
    두 모델은 각각 가격 변동 방향과 변동폭을 예측하고, 이를 결합하여 최종 예측 결과로써 사용합니다.

    Attributes
    ----------
    xgb_model : XGBClassifier
        가격 변동 방향을 예측하는 XGBoost 기본 모델.
    rfr_model : RandomForestRegressor
        가격 변동폭을 예측하는 RandomForest 기본 모델.

    Methods
    -------
    fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series)
        주어진 데이터를 사용하여 앙상블 모델을 학습합니다.
    predict(self, X: pd.DataFrame)
        학습된 모델을 기반으로 타겟 값을 예측합니다.
    """

    def preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """입력 데이터 X에 대해 전처리를 수행합니다. NaN, 무한대, 너무 큰 값을 처리합니다.

        Parameters
        ----------
        X : pd.DataFrame
            입력 특성 데이터입니다.

        Returns
        -------
        pd.DataFrame
            전처리된 데이터를 반환합니다.
        """
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())

        X = X.astype(np.float32)

        max_float32 = np.finfo(np.float32).max
        X[X > max_float32] = max_float32
        
        return X

    def __init__(self):
        """
        XGBoost, RandomForest 모델을 초기화합니다.
        """
        self.xgb_model = XGBClassifier()
        self.rfr_model = RandomForestRegressor()

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        X = self.preprocess_X(X)

        y_magnitude = y_price.pct_change().fillna(0).copy() * 100
        y_direction = y_magnitude.apply(lambda x: 1 if x > 0 else 0)

        X_train_dir, X_val_dir, y_train_dir, y_val_dir = train_test_split(
            X, y_direction, test_size=0.2, random_state=42
        )
        X_train_mag, X_val_mag, y_train_mag, y_val_mag = train_test_split(
            X, y_magnitude, test_size=0.2, random_state=42
        )

        self.xgb_model.fit(X_train_dir, y_train_dir)
        self.rfr_model.fit(X_train_mag, y_train_mag)

        self.X_val_dir, self.X_val_mag = X_val_dir, X_val_mag
        self.y_val_dir, self.y_val_mag = y_val_dir, y_val_mag

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        학습된 모델을 사용하여 검증 세트에 대해 타겟 값을 예측합니다.

        Parameters
        ----------
        X : pd.DataFrame
            입력 데이터입니다.

        Returns
        -------
        pd.Series
            예측된 타겟 값을 반환합니다.
        """
        X = self.preprocess_X(X)

        pred_direction = self.xgb_model.predict(X)
        pred_magnitude = self.rfr_model.predict(X)

        def price_movement_output(magnitude: float) -> int:
            """
            가격 변동폭을 두 가지 범주로 분류합니다.

            Parameters
            ----------
            magnitude : float
                가격 변동폭을 나타내는 실수 값입니다.

            Returns
            -------
            int
                가격 변동폭을 두 가지 범주로 분류한 결과값을 반환합니다.
                - 0: 변동폭의 절대값이 0.5 미만인 경우
                - 1: 변동폭의 절대값이 0.5 이상인 경우
            """
            if abs(magnitude) >= 0.5:
                return 1
            else:
                return 0
            
        def combined_price_prediction(direction: int, magnitude_category: int) -> int:
            """
            가격 변동 방향과 변동폭을 결합하여 4개의 카테고리로 분류합니다.

            Parameters
            ----------
            direction : int
                가격 변동 방향을 나타내는 값입니다. 
                1이면 가격 상승을 의미하고, 0이면 가격 하락을 의미합니다.
        
            magnitude_category : int
                가격 변동폭의 크기를 나타내는 범주 값입니다. 
                1이면 큰 변동, 0이면 작은 변동을 의미합니다.

            Returns
            -------
            int
                가격 변동 방향과 변동폭을 결합하여 4개의 카테고리로 분류한 결과값을 반환합니다.
                - 0: 가격 하락(0) + 큰 변동(1)
                - 1: 가격 하락(0) + 작은 변동(0)
                - 2: 가격 상승(1) + 작은 변동(0)
                - 3: 가격 상승(1) + 큰 변동(1)
            """
            if direction == 1:
                return 3 if magnitude_category == 1 else 2
            else:
                return 0 if magnitude_category == 1 else 1

        final_prediction = np.array([
            combined_price_prediction(direction, price_movement_output(magnitude))
            for direction, magnitude in zip(pred_direction, pred_magnitude)
        ])
        y_predict = pd.Series(final_prediction)

        return y_predict




       