import logging
import time

import pandas as pd

from .data_processors import BaseProcessor

logger = logging.getLogger(__name__)


class DataTransformPipeline:
    def __init__(self, steps: list[BaseProcessor]) -> None:
        self.steps = steps

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Запуск DataTransformPipeline. Всего шагов: {len(self.steps)}.")
        
        try:
            X_transformed = X.copy()
        except Exception as e:
            logger.error(f"Ошибка при копировании исходного датафрейма: {e}")
            raise
            
        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logger.info(f"Шаг {i}/{len(self.steps)}: Запуск {step_name}...")
            
            try:
                start_time = time.time()
                
                # Применяем трансформацию
                X_transformed = step.fit_transform(X_transformed)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Шаг {i} завершен за {elapsed_time:.2f} сек. Форма данных: {X_transformed.shape}")
                
            except Exception as e:
                # Если какой-то процессор падает, мы логируем его имя и прерываем весь конвейер
                logger.error(f"Критическая ошибка на шаге {i} ({step_name}). Пайплайн остановлен. Детали: {e}")
                raise
                
        logger.info("DataTransformPipeline успешно завершил обработку всех данных.")
        return X_transformed