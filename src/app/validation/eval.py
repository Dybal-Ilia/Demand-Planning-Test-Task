import numpy as np
import pandas as pd


class ForecastEvaluator:
    """
    Модуль для проведения Out-of-Time валидации и расчета метрик.
    Отрезает последние N периодов для теста, обучает оркестратор на трейне 
    и сравнивает прогнозы с фактом.
    """
    def __init__(self, period_limit: int = 36):
        self.period_limit = period_limit

    def evaluate(self, df: pd.DataFrame, orchestrator) -> pd.DataFrame:
        df_work = df.copy()
        df_work['period'] = pd.to_datetime(df_work['period'])
        
        # 1. Ищем точку отсечения (сплит на Train и Test)
        unique_dates = sorted(df_work['period'].unique())
        if len(unique_dates) <= self.period_limit:
            raise ValueError("Недостаточно истории для валидации. Данных меньше, чем размер горизонта.")
            
        split_date = unique_dates[-self.period_limit]
        
        train_df = df_work[df_work['period'] < split_date]
        test_df = df_work[df_work['period'] >= split_date]
        
        # 2. Вызываем оркестратор на усеченных данных (train_df)
        # Оркестратор сам поймет, что история закончилась раньше, и спрогнозирует
        # ровно те месяцы, которые лежат у нас в test_df.
        forecasts = orchestrator.fit_predict(train_df)
        forecasts['period'] = pd.to_datetime(forecasts['period'])
        
        # 3. Объединяем прогноз с фактическими продажами (true_demand)
        merged = pd.merge(
            forecasts,
            test_df[['sku', 'period', 'true_demand']],
            on=['sku', 'period'],
            how='inner'
        )
        
        # 4. Рассчитываем метрики в разрезе моделей/ABC-классов
        return self._calculate_metrics(merged)

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        # Группируем по моделям, чтобы сравнить нашу умную маршрутизацию с бейзлайном
        for abc_class, group in df.groupby('abc_class'):
            model_name = group['model_used'].iloc[0]
            
            y_true = group['true_demand'].values
            y_pred = group['forecast'].values
            y_base = group['baseline_forecast'].values
            
            # Считаем метрики для основной модели
            model_metrics = self._calc_error(y_true, y_pred)
            
            # Считаем метрики для наивного бейзлайна
            base_metrics = self._calc_error(y_true, y_base)
            
            results.append({
                'abc_class': abc_class,
                'model': model_name,
                
                'model_WAPE': model_metrics['wape'],
                'base_WAPE': base_metrics['wape'],
                
                'model_sMAPE': model_metrics['smape'],
                'base_sMAPE': base_metrics['smape'],
                
                'model_RMSE': model_metrics['rmse'],
                'base_RMSE': base_metrics['rmse']
            })
            
        return pd.DataFrame(results).round(3)

    def _calc_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Внутренняя функция для расчета математических формул"""
        # RMSE (Среднеквадратичная ошибка)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # WAPE (Взвешенная абсолютная ошибка в процентах) - стандарт ритейла
        sum_actual = np.sum(y_true)
        if sum_actual == 0:
            wape = np.nan
        else:
            wape = np.sum(np.abs(y_true - y_pred)) / sum_actual
            
        # sMAPE (Симметричная средняя абсолютная ошибка в процентах)
        denominator = np.abs(y_true) + np.abs(y_pred)
        # Защита от деления на ноль там, где факт и прогноз равны 0
        smape = np.mean(
            np.where(denominator == 0, 0, 2 * np.abs(y_true - y_pred) / denominator)
        )
        
        return {'rmse': rmse, 'wape': wape, 'smape': smape}