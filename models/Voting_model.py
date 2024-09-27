import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from abc import ABCMeta, abstractmethod
from model import Model

# CustomModel 클래스 정의 (y_price 없이)
class CustomModel(Model):
    """
    LightGBM과 XGBoost를 사용한 투표 분류기와 RFE 기반 피처 선택을 포함하는 커스텀 모델입니다.
    또한 0과 3, 1과 2 그룹에 대해 2단계 예측을 수행합니다.

    속성
    -----
    voting_model : VotingClassifier
        LightGBM과 XGBoost 모델을 결합한 투표 분류기입니다.
    voting_model_12 : VotingClassifier
        1과 2 그룹을 위한 투표 분류기입니다.
    best_rfe : RFE
        GridSearchCV를 통해 찾은 최적의 피처 선택 모델입니다.
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
        CustomModel 클래스의 초기화 메서드입니다.
        """
        self.voting_model = None
        self.voting_model_12 = None
        self.best_rfe = None
        self.best_stage2_model_03 = None

    def fit(self, X: pd.DataFrame, y: pd.Series, y_price: pd.Series) -> None:
        """
        피처 선택과 투표 분류기 학습을 포함하여 커스텀 모델을 학습합니다.

        매개변수
        ----------
        X : pd.DataFrame
            학습에 사용될 입력 데이터입니다.
        y : pd.Series
            입력 데이터에 대한 타겟 값입니다.
        """
        
        param_grid_rfe = {'n_features_to_select': [5, 10, 15]}
        rfe_model = LGBMClassifier(random_state=42, boost_from_average=False)
        rfe_search = GridSearchCV(
            estimator=RFE(estimator=rfe_model, step=1),
            param_grid=param_grid_rfe,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        rfe_search.fit(X, y)
        self.best_rfe = rfe_search.best_estimator_
        X_selected = self.best_rfe.transform(X)

        # 클래스 가중치 계산
        class_counts = Counter(y)
        class_weights = {cls: count for cls, count in class_counts.items()}

        # 파라미터 그리드 설정
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50],
            'max_depth': [5, 10, 20],
            'lambda_l1': [0.1, 1, 5, 10],
            'lambda_l2': [0.1, 1, 5, 10, 20],
            'min_child_samples': [20, 50, 100],
            'boost_from_average': [False],
            'class_weight': [class_weights]
        }

        # LightGBM 모델에 대한 Grid Search
        grid_search = GridSearchCV(
            estimator=LGBMClassifier(random_state=42, class_weight='balanced', reg_alpha=0.0, reg_lambda=0.0),
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X, y)

        # Voting Classifier 생성
        xgb_model = XGBClassifier(random_state=42)
        self.voting_model = VotingClassifier(
            estimators=[('lgbm', grid_search.best_estimator_), ('xgb', xgb_model)],
            voting='soft'
        )

        # Voting Classifier 학습
        self.voting_model.fit(X_selected, y)

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
        rfe_model = LGBMClassifier(random_state=42, boost_from_average=False)
        rfe_search_03 = GridSearchCV(
            estimator=RFE(estimator=rfe_model, step=1),
            param_grid={'n_features_to_select': [5, 10, 15]},
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        rfe_search_03.fit(X, y)
        best_rfe_03 = rfe_search_03.best_estimator_
        X_selected_03 = best_rfe_03.transform(X)

        self.best_stage2_model_03 = LGBMClassifier(random_state=42, boost_from_average=False)
        self.best_stage2_model_03.fit(X_selected_03, y)

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
        rfe_model = LGBMClassifier(random_state=42, boost_from_average=False)
        rfe_search_12 = GridSearchCV(
            estimator=RFE(estimator=rfe_model, step=1),
            param_grid={'n_features_to_select': [5, 10, 15]},
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        rfe_search_12.fit(X, y)
        best_rfe_12 = rfe_search_12.best_estimator_
        X_selected_12 = best_rfe_12.transform(X)

        lgbm_random = RandomizedSearchCV(
            LGBMClassifier(random_state=42, boost_from_average=False),
            param_distributions={'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
            cv=3, scoring='accuracy', n_iter=10, random_state=42
        )
        xgb_random = RandomizedSearchCV(
            XGBClassifier(random_state=42),
            param_distributions={'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
            cv=3, scoring='accuracy', n_iter=10, random_state=42
        )

        lgbm_random.fit(X_selected_12, y)
        xgb_random.fit(X_selected_12, y)

        best_lgbm = lgbm_random.best_estimator_
        best_xgb = xgb_random.best_estimator_

        self.voting_model_12 = VotingClassifier(
            estimators=[('lgbm', best_lgbm), ('xgb', best_xgb)],
            voting='soft'
        )
        self.voting_model_12.fit(X_selected_12, y)

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
        X_selected = self.best_rfe.transform(X)
        test_pred_stage1 = self.voting_model.predict(X_selected)

        test_df_03 = X[test_pred_stage1 == 1]
        if not test_df_03.empty:
            test_pred_03 = self.best_stage2_model_03.predict(test_df_03)
        else:
            test_pred_03 = np.array([])

        test_df_12 = X[test_pred_stage1 == 0]
        if not test_df_12.empty:
            test_pred_12 = self.voting_model_12.predict(test_df_12)
            test_pred_12 = np.where(test_pred_12 == 0, 1, 2)
        else:
            test_pred_12 = np.array([])

        final_test_pred = np.zeros(len(X))
        final_test_pred[test_pred_stage1 == 1] = test_pred_03
        final_test_pred[test_pred_stage1 == 0] = test_pred_12

        # 최종 예측 값을 정수형으로 변환
        final_test_pred = final_test_pred.astype(int)

        return pd.Series(final_test_pred)
