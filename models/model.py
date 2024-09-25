from abc import ABCMeta, abstractmethod
import pandas as pd

class Model(metaclass=ABCMeta):

    @abstractmethod
    def fit(X: pd.DataFrame, y: pd.Series) -> None:
        """가격 변동 예측 모델을 학습시킵니다.

        Parameters
        ----------
        X : pd.DataFrame
            마켓 및 네트워크 정보가 담겨져 있는 학습데이터입니다.
            
        y : pd.Series
            이후 가격 변동을 나타내는 학습데이터의 라벨입니다.

            0: -0.5% 미만
            1: -0.5% ~ 0% 
            2: 0% ~ 0.5%
            3: 0.5% 이상
        """

    @abstractmethod
    def predict(X: pd.DataFrame) -> pd.Series:
        """이후 가격 변동을 예측합니다.

        메서드를 호출하기 전, fit 메서드를 먼저 호출해야합니다.

        Parameters
        ----------
        X : pd.DataFrame
            마켓 및 네트워크 정보가 담겨져 있는 테스트데이터입니다.

        Returns
        -------
        pd.Series
            예측한 테스트데이터의 라벨입니다.
            
            0: -0.5% 미만
            1: -0.5% ~ 0% 
            2: 0% ~ 0.5%
            3: 0.5% 이상
        """
