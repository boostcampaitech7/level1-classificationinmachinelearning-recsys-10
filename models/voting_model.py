import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from model import Model



class VotingModel(Model):
    """단순 분류 모델을 사용하여 주가 등락을 예측하는 모델입니다."""

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
            }
        if selected_features is None:
            selected_features = 'all'

        self.model_params = model_params
        self.selected_features = selected_features
        self.ignore_strength = ignore_strength

        # 모델 초기화
        self.voting_model_stage1 = None
        self.lgbm_model_stage2_03 = None
        self.voting_model_stage2_12 = None

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        """모델 학습을 위한 함수"""

        # NaN 및 inf 값을 처리하기 위한 함수
        def _check_and_handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
            """NaN 및 inf 값을 처리"""
            df = df.replace([np.inf, -np.inf], np.nan)  # inf 값을 NaN으로 대체
            df = df.fillna(0)  # NaN 값을 0으로 대체 (필요에 따라 다른 값으로 대체 가능)
            return df
        
        # NaN 및 inf 값 처리
        X = self._check_and_handle_invalid_values(X)

        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]

        if self.ignore_strength:
            y = y.apply(lambda x: 1 if x <= 1 else 2)

        

        # Stage 1 모델 학습 (0,3 vs 1,2)
        lgbm_model = LGBMClassifier(
            learning_rate=0.1,
            num_leaves=50,
            max_depth=5,
            lambda_l1=0.1,
            lambda_l2=0.1,
            min_child_samples=100,
            boost_from_average=False,
            random_state=42
        )

        xgb_model = XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        self.voting_model_stage1 = VotingClassifier(
            estimators=[
                ('lgbm', lgbm_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )

        self.voting_model_stage1.fit(selected_X, y)

        # Stage 2: 0 vs 3 모델 학습
        group_03 = X[y == 1]
        y_03 = y[y == 1].apply(lambda x: 0 if x == 0 else 3)
        self.lgbm_model_stage2_03 = LGBMClassifier(random_state=42, boost_from_average=False)
        self.lgbm_model_stage2_03.fit(group_03, y_03)

        # Stage 2: 1 vs 2 모델 학습
        group_12 = X[y == 0]
        y_12 = y[y == 0].apply(lambda x: 1 if x == 1 else 2)

        lgbm_model_12 = LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            boost_from_average=False
        )

        xgb_model_12 = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

        self.voting_model_stage2_12 = VotingClassifier(
            estimators=[('lgbm', lgbm_model_12), ('xgb', xgb_model_12)],
            voting='soft'
        )
        self.voting_model_stage2_12.fit(group_12, y_12)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """테스트 데이터에 대한 예측을 수행"""
        if self.selected_features == 'all':
            selected_X = X
        else:
            selected_X = X[self.selected_features]

        # 1단계 예측 (0,3 vs 1,2)
        test_pred_stage1 = self.voting_model_stage1.predict(selected_X)

        # 0과 3 예측
        test_df_03 = selected_X[test_pred_stage1 == 1]
        if not test_df_03.empty:
            test_pred_03 = self.lgbm_model_stage2_03.predict(test_df_03)
        else:
            test_pred_03 = np.array([])

        # 1과 2 예측
        test_df_12 = selected_X[test_pred_stage1 == 0]
        if not test_df_12.empty:
            test_pred_12 = self.voting_model_stage2_12.predict(test_df_12)
            test_pred_12 = np.where(test_pred_12 == 0, 1, 2)  # 0을 1로, 1을 2로 변경
        else:
            test_pred_12 = np.array([])

        # 최종 결과 결합
        final_test_pred = np.zeros(len(selected_X))  # 기본적으로 모두 0으로 설정
        final_test_pred[test_pred_stage1 == 1] = test_pred_03  # 0, 3의 예측 결과
        final_test_pred[test_pred_stage1 == 0] = test_pred_12  # 1, 2의 예측 결과

        return pd.Series(final_test_pred).astype(int)
