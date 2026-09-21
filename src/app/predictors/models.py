import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.Series):
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        pass

class NaiveBaselineForecaster(BaseModel):
    def __init__(self):
        self.last_value = 0

    def fit(self, X: pd.Series):
        clean_X = X.dropna()
        if not clean_X.empty:
            self.last_value = clean_X.iloc[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(shape=horizon, fill_value=self.last_value)

class MovingAverageForecaster(BaseModel):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.history = None

    def fit(self, y: pd.Series):
        clean_y = y.dropna().astype(float)
        if len(clean_y) < self.window_size:
            pad_size = self.window_size - len(clean_y)
            self.history = np.pad(clean_y.values, (pad_size, 0), 'constant', constant_values=0)
        else:
            self.history = clean_y.iloc[-self.window_size:].values
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.history is None:
            raise ValueError("Модель не обучена")
            
        predictions = []
        current_window = list(self.history)
        
        for _ in range(horizon):
            next_val = np.mean(current_window)
            predictions.append(next_val)
            current_window.pop(0)
            current_window.append(next_val)
            
        return np.array(predictions)

class StatsmodelsBaseForecaster(BaseModel):
    """Базовый класс, скрывающий бойлерплейт обработки ошибок statsmodels."""
    def __init__(self):
        self.model_fit = None
        self.fallback_value = 0

    def _safe_fit_predict(self, y: pd.Series, model_builder_func):
        clean_y = y.dropna().astype(float)
        if not clean_y.empty:
            self.fallback_value = clean_y.iloc[-1]
            
        try:
            model = model_builder_func(clean_y)
            if model is not None:
                self.model_fit = model.fit(optimized=True)
        # Ловим математические исключения statsmodels
        except (ValueError, TypeError, NotImplementedError, ZeroDivisionError) as e:
            logger.warning(f"{self.__class__.__name__}: Сбой оптимизатора, включен fallback. Ошибка: {e}")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.model_fit is None:
            return np.full(shape=horizon, fill_value=self.fallback_value)
        try:
            return self.model_fit.forecast(horizon).values
        except (ValueError, TypeError) as e:
            logger.error(f"{self.__class__.__name__}: Ошибка прогноза. Ошибка: {e}")
            return np.full(shape=horizon, fill_value=self.fallback_value)

class SESForecaster(StatsmodelsBaseForecaster):
    def fit(self, y: pd.Series):
        def builder(clean_y):
            return SimpleExpSmoothing(clean_y, initialization_method="estimated") if len(clean_y) >= 3 else None
        return self._safe_fit_predict(y, builder)

class HoltWintersForecaster(StatsmodelsBaseForecaster):
    def __init__(self, seasonal_periods: int = 12):
        super().__init__()
        self.seasonal_periods = seasonal_periods

    def fit(self, y: pd.Series):
        def builder(clean_y):
            if len(clean_y) < self.seasonal_periods * 2:
                return None
            return ExponentialSmoothing(
                clean_y, trend='add', seasonal='add', 
                seasonal_periods=self.seasonal_periods, initialization_method="estimated"
            )
        return self._safe_fit_predict(y, builder)