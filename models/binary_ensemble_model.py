from model import Model
import numpy as np
import pandas as pd
import lightgbm as lgb


class BinaryEnsembleModel(Model):
    """다중 클래스를 이진 분류 모델로 예측하는 앙상블 모델입니다."""

    def __init__(self, model_params: dict = None, selected_features: list = None):
        """분류 모델의 각종 설정을 초기화합니다.

        Parameters
        ----------
        model_params : dict, optional
            lightgbm.train에 들어가는 파라미터들을 정의한 딕셔너리입니다.
            전달하지 않은 경우 lightgbm의 default hyperparameter를 사용합니다.
        selected_features : list, optional
            사용할 feature들의 이름을 담은 리스트입니다.
            전달하지 않은 경우 모든 feature를 사용합니다.
        """
        if model_params is None:
            model_params = {
                "random_state": 42,
                "verbose": -1,
            }
        if selected_features is None:
            selected_features = "all"

        self.model_params = model_params
        self.selected_features = selected_features
        self.binary_models = {}  # 각 클래스에 대한 이진 분류 모델 저장용

    def fit(self, X: pd.DataFrame, y: pd.Series, category_cols: list = None) -> None:
        """각 클래스별 이진 분류 모델을 학습합니다."""
        if self.selected_features == "all":
            selected_X = X
        else:
            selected_X = X[self.selected_features]

        classes = [0, 1, 2, 3]  # 분류할 클래스 목록

        for target_class in classes:
            # 이진 타겟 생성
            y_train_binary = (y == target_class).astype(int)

            # LightGBM 데이터셋 생성
            train_data = lgb.Dataset(
                selected_X, label=y_train_binary, categorical_feature=category_cols
            )

            # 모델 학습
            model = lgb.train(
                self.model_params,
                train_set=train_data,
                num_boost_round=600,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(100),
                ],
            )

            # 학습된 모델을 클래스별로 저장
            self.binary_models[target_class] = model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """앙상블을 통해 다중 클래스 예측을 수행합니다."""
        if self.selected_features == "all":
            selected_X = X
        else:
            selected_X = X[self.selected_features]

        predictions = np.zeros((selected_X.shape[0], len(self.binary_models)))

        # 각 클래스에 대해 예측 확률을 계산
        for i, (target_class, model) in enumerate(self.binary_models.items()):
            pred_proba = model.predict(selected_X, num_iteration=model.best_iteration)
            predictions[:, i] = pred_proba

        # 가장 높은 확률의 클래스를 예측
        final_predictions = np.argmax(predictions, axis=1)
        return pd.Series(final_predictions)
