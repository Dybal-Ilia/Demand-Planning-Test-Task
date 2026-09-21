import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Базовый класс для всех обработчиков данных в пайплайне.
    Реализует паттерн проектирования, схожий с API scikit-learn.
    """
    def fit(self, X: pd.DataFrame, y=None):
        """
        Обучает трансформер (вычисляет и сохраняет статистики из данных).
        Дефолтная реализация возвращает self для процессоров без состояния.
        
        Args:
            X (pd.DataFrame): Входной датафрейм для обучения.
            y: Игнорируется, присутствует для совместимости API.
            
        Returns:
            self: Обученный инстанс процессора.
        """
        return self # Дефолтная реализация для тех, кому не нужно обучаться

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет трансформацию к переданным данным.
        
        Args:
            X (pd.DataFrame): Исходные данные.
            
        Returns:
            pd.DataFrame: Преобразованные данные.
        """

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Последовательно вызывает fit и transform.
        """
        return self.fit(X, y).transform(X)


class DropDuplicatesProcessor(BaseProcessor):
    """
    Очистка данных от полных дубликатов на уровне (Товар, Локация, Период).
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            initial_rows = len(X)
            df_clean = X.drop_duplicates(subset=["sku", "location", "period"]).copy()
            dropped = initial_rows - len(df_clean)
            if dropped > 0:
                logger.info(f"DropDuplicates: Удалено {dropped} дублирующих записей.")
            return df_clean
        except KeyError as e:
            logger.error(f"DropDuplicates: Отсутствуют обязательные колонки для дедупликации: {e}")
            raise


class NegativeSellsProcessor(BaseProcessor):
    """
    Обработка отрицательных продаж.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            neg_mask = df['qty'] < 0
            neg_count = neg_mask.sum()
            
            if neg_count > 0:
                df.loc[neg_mask, 'qty'] = 0
                logger.info(f"NegativeSells: Обнулено {neg_count} записей с отрицательными продажами.")
            return df
        except KeyError:
            logger.error("NegativeSells: Не найдена колонка 'qty'.")
            raise


class UOMCorrectionProcessor(BaseProcessor):
    """
    Детектирование и исправление аномальных скачков размерности (Unit of Measure).
    """
    def __init__(self, window_size=3, threshold=50):
        self.window_size = window_size
        self.threshold = threshold
        self.corrections_ = {} 

    def fit(self, X: pd.DataFrame, y=None):
        try:
            df = X.copy()
            df['period'] = pd.to_datetime(df['period'])
            df = df.sort_values(['sku', 'period'])
            
            self.corrections_ = {}
            
            for sku, group in df.groupby('sku'):
                ts = group.set_index('period')['qty']
                
                med_before = ts.rolling(window=self.window_size, min_periods=1).median()
                med_after = ts.iloc[::-1].rolling(window=self.window_size, min_periods=1).median().iloc[::-1]
                
                med_after_shifted = med_after.shift(-1)
                ratio = med_after_shifted / (med_before + 1)
                
                anomalies = ratio[ratio > self.threshold]
                
                if not anomalies.empty:
                    shift_date = anomalies.index[0] + pd.DateOffset(months=1)
                    actual_ratio = anomalies.iloc[0]
                    scale = 10 ** int(np.round(np.log10(actual_ratio)))
                    
                    self.corrections_[sku] = {
                        'shift_date': shift_date,
                        'scale': scale
                    }
                    
            if self.corrections_:
                logger.info(f"UOMCorrection (fit): Найдено {len(self.corrections_)} SKU со сдвигами размерности.")
            return self
            
        except Exception as e:
            logger.error(f"UOMCorrection (fit): Ошибка при расчете сдвигов UOM. Детали: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X_transformed = X.copy()
            X_transformed['qty'] = X_transformed['qty'].astype(float)
            
            is_string_period = pd.api.types.is_string_dtype(X_transformed['period'])
            if is_string_period:
                X_transformed['period_dt'] = pd.to_datetime(X_transformed['period'])
            else:
                X_transformed['period_dt'] = X_transformed['period']

            applied_count = 0
            for sku, params in self.corrections_.items():
                mask = (X_transformed['sku'] == sku) & (X_transformed['period_dt'] >= params['shift_date'])
                affected_rows = mask.sum()
                if affected_rows > 0:
                    X_transformed.loc[mask, 'qty'] = X_transformed.loc[mask, 'qty'] / params['scale']
                    applied_count += 1
            
            if is_string_period:
                X_transformed = X_transformed.drop(columns=['period_dt'])
                
            if applied_count > 0:
                logger.info(f"UOMCorrection (transform): Скорректированы объемы для {applied_count} SKU.")
            return X_transformed
            
        except Exception as e:
            logger.error(f"UOMCorrection (transform): Ошибка применения корректировок UOM. Детали: {e}")
            raise


class NetworkAggregationProcessor(BaseProcessor):
    """
    Агрегация временных рядов с уровня конкретного склада (Location) на уровень всей сети.
    """
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            initial_rows = len(df)
            
            agg_funcs = {
                'qty': 'sum',
                'stock_end_qty': 'sum',
                'days_out_of_stock': 'max'
            }
            
            static_cols = ['category', 'abc_class', 'launch_period', 'uom', 'predecessor_sku']
            for col in static_cols:
                if col in df.columns:
                    agg_funcs[col] = 'first'
                    
            df_agg = df.groupby(['sku', 'period'], as_index=False).agg(agg_funcs)
            logger.info(f"NetworkAggregation: Данные агрегированы. Строк до: {initial_rows}, после: {len(df_agg)}.")
            return df_agg
            
        except Exception as e:
            logger.error(f"NetworkAggregation: Ошибка при агрегации сети. Убедитесь в наличии числовых колонок. Детали: {e}")
            raise


class TimeGridProcessor(BaseProcessor):
    """
    Построение непрерывной временной сетки (Cartesian Product).
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            df['period'] = pd.to_datetime(df['period'])
            
            unique_skus = df['sku'].unique()
            date_range = pd.date_range(
                start=df['period'].min(), 
                end=df['period'].max(), 
                freq='MS'
            )
            
            grid = pd.MultiIndex.from_product(
                [unique_skus, date_range], 
                names=['sku', 'period']
            ).to_frame(index=False)
            
            df_grid = pd.merge(grid, df, on=['sku', 'period'], how='left')
            
            fill_zero_cols = ['qty', 'stock_end_qty', 'days_out_of_stock']
            for col in fill_zero_cols:
                if col in df_grid.columns:
                    df_grid[col] = df_grid[col].fillna(0)
                
            static_cols = ['category', 'abc_class', 'launch_period', 'uom', 'predecessor_sku']
            metadata = X.dropna(subset=['category']).drop_duplicates('sku').set_index('sku')
            
            for col in static_cols:
                if col in df_grid.columns:
                    df_grid[col] = df_grid['sku'].map(metadata[col])
                    
            df_final = df_grid.sort_values(['sku', 'period']).reset_index(drop=True)
            logger.info(f"TimeGrid: Сетка построена. Добавлено {len(df_final) - len(df)} пустых периодов.")
            return df_final
            
        except Exception as e:
            logger.error(f"TimeGrid: Ошибка при построении временной сетки. Детали: {e}")
            raise


class OOSRestorationProcessor(BaseProcessor):
    """
    Восстановление истинного спроса (True Demand) в периоды дефицита (Out-of-Stock).
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['period']):
                df['period'] = pd.to_datetime(df['period'])
                
            days_in_month = df['period'].dt.daysinmonth
            df['true_demand'] = df['qty'].astype(float)
            
            # Ситуация 1: Частичный OOS
            partial_oos = (df['days_out_of_stock'] > 0) & (df['days_out_of_stock'] <= days_in_month - 3)
            days_on_shelf = days_in_month[partial_oos] - df.loc[partial_oos, 'days_out_of_stock']
            df.loc[partial_oos, 'true_demand'] = (
                df.loc[partial_oos, 'qty'] / days_on_shelf * days_in_month[partial_oos]
            )
            
            # Ситуация 2: Полный OOS
            full_oos = df['days_out_of_stock'] > days_in_month - 3
            rolling_avg = df.groupby('sku')['true_demand'].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
            df.loc[full_oos, 'true_demand'] = rolling_avg[full_oos]
            
            # Защита от краевых случаев
            df['true_demand'] = df.groupby('sku')['true_demand'].transform(lambda x: x.bfill().fillna(0))
            df['true_demand'] = df['true_demand'].round().astype(int)
            
            logger.info(f"OOSRestoration: Восстановлен спрос для {partial_oos.sum()} частичных и {full_oos.sum()} полных OOS-периодов.")
            return df
            
        except Exception as e:
            logger.error(f"OOSRestoration: Ошибка логики восстановления Out-of-Stock. Детали: {e}")
            raise


class PromoFeatureProcessor(BaseProcessor):
    """
    Генерация признаков маркетинговых активностей.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        try:
            if 'promo_type' in df.columns:
                df['promo_type'] = df['promo_type'].fillna('No_Promo')
                df['is_promo'] = (df['promo_type'] != 'No_Promo').astype(int)
                logger.info("PromoFeature: Флаг 'is_promo' успешно создан.")
            else:
                logger.warning("PromoFeature: Колонка 'promo_type' не найдена. Создание 'is_promo' пропущено.")
                
            if 'discount_pct' in df.columns:
                df['discount_pct'] = df['discount_pct'].fillna(0.0)
                
            return df
        except Exception as e:
            logger.error(f"PromoFeature: Ошибка при обработке промо-признаков. Детали: {e}")
            raise


class PredecessorMergeProcessor(BaseProcessor):
    """
    Слияние истории продаж товаров-предшественников.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df = X.copy()
            if 'predecessor_sku' not in df.columns:
                logger.warning("PredecessorMerge: Колонка 'predecessor_sku' отсутствует. Слияние пропущено.")
                return df
                
            mapping = df[['sku', 'predecessor_sku']].dropna().drop_duplicates()
            skus_to_drop = []
            
            for _, row in mapping.iterrows():
                new_sku = row['sku']
                old_sku = row['predecessor_sku']
                
                old_history = df[df['sku'] == old_sku].copy()
                
                if not old_history.empty:
                    new_sku_first_period = df[df['sku'] == new_sku]['period'].min()
                    old_history = old_history[old_history['period'] < new_sku_first_period]
                    
                    new_sku_metadata = df[df['sku'] == new_sku].iloc[0]
                    old_history['sku'] = new_sku
                    
                    static_cols = ['category', 'abc_class', 'launch_period', 'predecessor_sku', 'uom']
                    for col in static_cols:
                        if col in old_history.columns:
                            old_history[col] = new_sku_metadata[col]
                    
                    skus_to_drop.append(old_sku)
                    df = pd.concat([df, old_history], ignore_index=True)
                    
            if skus_to_drop:
                df = df[~df['sku'].isin(skus_to_drop)]
                
            logger.info(f"PredecessorMerge: Подклеена история для {len(skus_to_drop)} предшественников. Старые SKU удалены.")
            return df.sort_values(['sku', 'period']).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"PredecessorMerge: Ошибка при объединении товаров-предшественников. Детали: {e}")
            raise