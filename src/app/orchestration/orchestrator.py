import logging

import numpy as np
import pandas as pd

from ..predictors.models import NaiveBaselineForecaster
from .registry import ForecastingRegistry

logger = logging.getLogger(__name__)


class ForecastingOrchestrator:
    """
    Класс-маршрутизатор. Распределяет товары по моделям в зависимости от ABC-класса,
    обучает их, генерирует прогноз и сравнивает его с Baseline.
    """
    def __init__(self, period_limit: int = 3):
        """
        :param period_limit: Горизонт прогнозирования в месяцах
        """
        self.period_limit = period_limit

    def fit_predict(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        try:
            df = clean_df.copy()
            df['period'] = pd.to_datetime(df['period'])
            
            last_date = df['period'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=self.period_limit, 
                freq='MS'
            )
            
            unique_skus = df['sku'].nunique()
            logger.info(f"ForecastingOrchestrator: Запуск для {unique_skus} SKU. Горизонт: {self.period_limit} мес.")
            
        except Exception as e:
            logger.error(f"ForecastingOrchestrator: Ошибка подготовки данных или расчета дат. Детали: {e}")
            raise

        success_count = 0
        error_count = 0

        for sku, group in df.groupby('sku'):
            sku_results, is_success = self._process_single_sku(sku, group, future_dates)
            results.extend(sku_results)
            
            if is_success:
                success_count += 1
            else:
                error_count += 1

        logger.info(f"ForecastingOrchestrator: Расчет завершен. Успешно: {success_count} SKU. С ошибками: {error_count} SKU.")
        
        return pd.DataFrame(results)

    def _process_single_sku(self, sku: str, group: pd.DataFrame, future_dates: pd.DatetimeIndex) -> tuple[list[dict], bool]:
        """
        Приватный метод для обработки и прогнозирования одного конкретного SKU.
        Возвращает список словарей (строк для датафрейма) и флаг успешности.
        """
        sku_results = []
        abc = 'C'  # Фолбэк на случай, если ниже что-то упадет при чтении класса
        
        try:
            group = group.sort_values('period')
            if 'abc_class' in group.columns:
                abc = group['abc_class'].iloc[0]
            
            model = ForecastingRegistry.get_model(abc)
            baseline = NaiveBaselineForecaster()
            
            y = group.set_index('period')['true_demand']
            
            model.fit(y)
            preds = model.predict(self.period_limit)
            
            baseline.fit(y)
            base_preds = baseline.predict(self.period_limit)
            
            for i, date in enumerate(future_dates):
                # Защита от NaN, если математика внутри модели сломается нетипично
                safe_pred = 0 if np.isnan(preds[i]) else preds[i]
                safe_base = 0 if np.isnan(base_preds[i]) else base_preds[i]

                sku_results.append({
                    'sku': sku,
                    'period': date.strftime('%Y-%m'),
                    'abc_class': abc,
                    'model_used': type(model).__name__,
                    'forecast': round(max(0, safe_pred)),
                    'baseline_forecast': round(max(0, safe_base))
                })
            
            return sku_results, True

        except Exception as e:  # noqa: BLE001
            # Защита оркестратора от падения на проблемном ряду
            logger.warning(f"Сбой при прогнозировании SKU {sku}. Заполняем нулями. Ошибка: {e}")
            
            for date in future_dates:
                sku_results.append({
                    'sku': sku,
                    'period': date.strftime('%Y-%m'),
                    'abc_class': abc,
                    'model_used': 'FallbackZero',
                    'forecast': 0,
                    'baseline_forecast': 0
                })
                
            return sku_results, False