import numpy as np
import pandas as pd
import pytest

from src.app.orchestration.registry import ForecastingRegistry, HoltWintersForecaster
from src.app.predictors.models import MovingAverageForecaster
from src.app.processors.data_processors import OOSRestorationProcessor


@pytest.fixture
def oos_sample_data() -> pd.DataFrame:
    """Фикстура генерирует тестовые данные для проверки Out-of-Stock логики"""
    return pd.DataFrame({
        'sku': ['SKU_TEST', 'SKU_TEST', 'SKU_TEST'],
        # Апрель (30 дней), Май (31 день), Июнь (30 дней)
        'period': pd.to_datetime(['2024-04-01', '2024-05-01', '2024-06-01']),
        'qty': [10.0, 100.0, 0.0],
        'days_out_of_stock': [20, 0, 30] 
    })

def test_oos_restoration_partial(oos_sample_data):
    """
    Проверяем частичный OOS.
    В апреле 30 дней. Товара не было 20 дней. На полке лежал 10 дней.
    Продали 10 штук. Истинный спрос за месяц должен быть: (10 / 10) * 30 = 30.
    """
    processor = OOSRestorationProcessor()
    res = processor.fit_transform(oos_sample_data)
    
    # Проверяем первый месяц (Апрель)
    assert res.loc[0, 'true_demand'] == 30

def test_oos_restoration_full(oos_sample_data):
    """
    Проверяем полный OOS.
    В июне товара не было все 30 дней. Продажи 0.
    Спрос должен восстановиться по среднему за предыдущие месяцы.
    Пред. месяцы (true_demand): Апрель = 30, Май = 100. Среднее = 65.
    """
    processor = OOSRestorationProcessor()
    res = processor.fit_transform(oos_sample_data)
    
    # Проверяем третий месяц (Июнь)
    assert res.loc[2, 'true_demand'] == 65

def test_moving_average_autoregression():
    """
    Проверяем, что скользящее среднее правильно "заглатывает" собственные 
    прогнозы при построении горизонта более 1 месяца.
    """
    forecaster = MovingAverageForecaster(window_size=3)
    y_train = pd.Series([10.0, 20.0, 30.0])
    
    forecaster.fit(y_train)
    preds = forecaster.predict(horizon=3)
    
    # Месяц 1: (10 + 20 + 30) / 3 = 20.0
    # Месяц 2: (20 + 30 + 20.0) / 3 = 23.333
    # Месяц 3: (30 + 20.0 + 23.333) / 3 = 24.444
    expected = np.array([20.0, 23.333, 24.444])
    
    # Используем np.allclose для сравнения дробных чисел с допуском
    assert np.allclose(preds, expected, atol=0.01)

def test_registry_routing():
    """
    Проверяем, что реестр выдает правильный инстанс модели по ABC классу,
    а при неизвестном классе отдает дефолтную (С).
    """
    model_a = ForecastingRegistry.get_model('A')
    model_c = ForecastingRegistry.get_model('C')
    model_unknown = ForecastingRegistry.get_model('XYZ') # Неизвестный класс
    
    assert isinstance(model_a, HoltWintersForecaster)
    assert isinstance(model_c, MovingAverageForecaster)
    # Фоллбэк должен вернуть MovingAverage
    assert isinstance(model_unknown, MovingAverageForecaster)