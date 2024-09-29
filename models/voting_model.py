import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from model import Model



class VotingModel(Model):
    """
    LightGBM과 XGBoost를 사용한 투표 분류기 커스텀 모델입니다.
    또한 0과 3, 1과 2 그룹에 대해 2단계 예측을 수행합니다.

    속성
    -----
    voting_model : VotingClassifier
        LightGBM과 XGBoost 모델을 결합한 투표 분류기입니다.
    voting_model_12 : VotingClassifier
        1과 2 그룹을 위한 투표 분류기입니다.
    best_stage2_model_03 : LGBMClassifier
        0과 3 그룹을 분류하는 모델입니다.

    메서드
    ------
    fit(X, y)
        피처 선택과 투표 분류기 학습을 포함한 모델 학습을 수행합니다.
    fit_group_03(X, y)
        0과 3 그룹 분류를 위한 모델을 학습합니다.
    fit_group_12(X, y)
        1과 2 그룹 분류를 위한 모델을 학습합니다.
    predict(X)
        학습된 모델을 사용하여 입력 데이터에 대한 예측을 수행합니다.
    """
    def __init__(self):
        """
        VotingModel 클래스의 초기화 메서드입니다.
        """
        self.voting_model = None
        self.voting_model_12 = None
        self.best_stage2_model_03 = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        피처 선택과 투표 분류기 학습을 포함하여 커스텀 모델을 학습합니다.

        매개변수
        ----------
        X : pd.DataFrame
            학습에 사용될 입력 데이터입니다.
        y : pd.Series
            입력 데이터에 대한 타겟 값입니다.
        """
        class_counts = Counter(y)
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        lgbm_model = LGBMClassifier(
            learning_rate=0.1,
            num_leaves=50,
            max_depth=5,
            lambda_l1=0.1,
            lambda_l2=0.1,
            min_child_samples=100,
            boost_from_average=False,
            class_weight=class_weights,
            random_state=42
        )

        xgb_model = XGBClassifier(
            learning_rate=0.1,  
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        self.voting_model = VotingClassifier(
            estimators=[
                ('lgbm', lgbm_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )

        # Voting Classifier 학습
        self.voting_model.fit(X, y)

        # 0과 3 그룹 분류 모델 학습
        group_03 = X[y == 1]
        y_train_03 = y.loc[group_03.index]
        self.fit_group_03(group_03, y_train_03)

        # 1과 2 그룹 분류 모델 학습
        group_12 = X[y == 0]
        y_train_12 = y.loc[group_12.index].apply(lambda x: 1 if x == 2 else 0)
        self.fit_group_12(group_12, y_train_12)

    def fit_group_03(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        0과 3 그룹 분류를 위한 모델을 학습합니다.

        매개변수
        ----------
        X : pd.DataFrame
            0과 3 그룹에 대한 입력 데이터입니다.
        y : pd.Series
            0과 3 그룹에 대한 타겟 값입니다.
        """
        self.best_stage2_model_03 = LGBMClassifier(random_state=42, boost_from_average=False)
        self.best_stage2_model_03.fit(X, y)

    def fit_group_12(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        1과 2 그룹 분류를 위한 모델을 학습합니다.

        매개변수
        ----------
        X : pd.DataFrame
            1과 2 그룹에 대한 입력 데이터입니다.
        y : pd.Series
            1과 2 그룹에 대한 타겟 값입니다.
        """
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

        self.voting_model_12 = VotingClassifier(
            estimators=[('lgbm', lgbm_model_12), ('xgb', xgb_model_12)],
            voting='soft'
        )

        self.voting_model_12.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        학습된 모델을 사용하여 입력 데이터에 대한 예측을 수행합니다.

        매개변수
        ----------
        X : pd.DataFrame
            예측을 수행할 입력 데이터입니다.

        반환값
        -------
        pd.Series
            입력 데이터에 대한 예측 값으로, 그룹 0, 1, 2, 3에 대한 결과를 반환합니다.
        """
        # 1단계 예측: 0,3 vs 1,2
        test_pred_stage1 = self.voting_model.predict(X)

        # 0과 3 예측
        test_df_03 = X[test_pred_stage1 == 1]
        if not test_df_03.empty:
            test_pred_03 = self.best_stage2_model_03.predict(test_df_03)
        else:
            test_pred_03 = np.array([])

        # 1과 2 예측
        test_df_12 = X[test_pred_stage1 == 0]
        if not test_df_12.empty:
            test_pred_12 = self.voting_model_12.predict(test_df_12)
            test_pred_12 = np.where(test_pred_12 == 0, 1, 2)
        else:
            test_pred_12 = np.array([])

        # 최종 결과 결합
        final_test_pred = np.zeros(len(X))
        final_test_pred[test_pred_stage1 == 1] = test_pred_03
        final_test_pred[test_pred_stage1 == 0] = test_pred_12

        return pd.Series(final_test_pred.astype(int))

