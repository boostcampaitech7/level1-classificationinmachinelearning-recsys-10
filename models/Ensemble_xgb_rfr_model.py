import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from model import Model

class Ensemble_XGB_RFR_Model(Model):
    """
    XGBoost와 RandomForestRegressor를 결합한 앙상블 모델.
    
    이 모델은 두 개의 레이어 앙상블 방식을 사용하여, 기본 모델(XGBoost와 RandomForestRegressor)의 예측을 결합하여 사용합니다.

    속성
    ----------
    model : LinearRegression
        기본 모델의 예측을 학습하는 메타 모델.
    xgb_model : XGBClassifier
        XGBoost 기본 모델.
    rfr_model : RandomForestRegressor
        RandomForest 기본 모델.

    메서드
    -------
    fit(X_direction, X_magnitude, y_direction, y_magnitude)
        주어진 데이터를 사용하여 앙상블 모델을 학습합니다.
    predict(X_direction, X_magnitude)
        학습된 모델을 기반으로 타겟 값을 예측합니다.
    """

    def __init__(self):
        """
        XGBoost, RandomForest을 메타 모델로 초기화합니다.
        """
        self.model = None
        self.xgb_model = XGBClassifier()
        self.rfr_model = RandomForestRegressor()

    def fit(self, X_direction: pd.DataFrame, X_magnitude: pd.DataFrame, 
            y_direction: pd.Series, y_magnitude: pd.Series, test_size: float = 0.2) -> None:
        """
        주어진 데이터셋에 앙상블 모델을 학습합니다.

        기본 모델(XGBoost와 RandomForest)은 train_test_split 검증을 사용하여 학습되며, 그 예측값을 결합하여 사용됩니다.

        매개변수
        ----------
        데이터를 두 개로 나눕니다: 가격 변동 방향 예측 (XGBClassifier)와 가격 변동폭 예측 (RandomForestRegressor)
        X_direction : pd.DataFrame
        가격 변동 방향을 예측하는 특성 행렬.
        X_magnitude : pd.DataFrame
        가격 변동폭을 예측하는 특성 행렬.
        y_direction : pd.Series
        가격 변동 방향에 대한 타겟 값.
        y_magnitude : pd.Series
        가격 변동폭에 대한 타겟 값.

        반환값
        -------
        None
        """
        X_train_dir, X_val_dir, y_train_dir, y_val_dir = train_test_split(
            X_direction, y_direction, test_size=test_size, random_state=42
        )
        X_train_mag, X_val_mag, y_train_mag, y_val_mag = train_test_split(
            X_magnitude, y_magnitude, test_size=test_size, random_state=42
        )

        self.xgb_model.fit(X_train_dir, y_train_dir)
        self.rfr_model.fit(X_train_mag, y_train_mag)

        self.X_val_dir, self.X_val_mag = X_val_dir, X_val_mag
        self.y_val_dir, self.y_val_mag = y_val_dir, y_val_mag

    def predict(self) -> np.ndarray:
        """
        학습된 모델을 사용하여 검증 세트에 대해 타겟 값을 예측합니다.

        반환값
        -------
        np.ndarray
            예측된 타겟 값.
        """
        pred_direction = self.xgb_model.predict(self.X_val_dir)
        pred_magnitude = self.rfr_model.predict(self.X_val_mag)

        def price_movement_output(magnitude):
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
            
        def combined_price_prediction(direction, magnitude_category):
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

        return final_prediction




       