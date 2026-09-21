import logging
import warnings

import pandas as pd

from src.app.orchestration.orchestrator import ForecastingOrchestrator
from src.app.processors.data_processors import (
    DropDuplicatesProcessor,
    NetworkAggregationProcessor,
    OOSRestorationProcessor,
    PredecessorMergeProcessor,
    PromoFeatureProcessor,
    TimeGridProcessor,
    UOMCorrectionProcessor,
)
from src.app.processors.pipeline import DataTransformPipeline
from src.app.validation.eval import ForecastEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    pipeline = DataTransformPipeline([
        DropDuplicatesProcessor(),
        PredecessorMergeProcessor(),
        UOMCorrectionProcessor(),
        NetworkAggregationProcessor(),
        TimeGridProcessor(),
        OOSRestorationProcessor(),
        PromoFeatureProcessor()
        ])
    full_df = pd.read_csv("./data/full_df.csv")
    clean_data = pipeline.fit_transform(full_df)


    period_limit_validation = 3
    period_limit_submit = 36

    logger.info("Запуск валидации моделей (Out-of-Time)")
    eval_orchestrator = ForecastingOrchestrator(period_limit=period_limit_validation)
    evaluator = ForecastEvaluator(period_limit=period_limit_validation)
    metrics_report = evaluator.evaluate(clean_data, eval_orchestrator)

    print(metrics_report.to_markdown(index=False))

    logger.info("Расчет финального прогноза на будущее")
    prod_orchestrator = ForecastingOrchestrator(period_limit=period_limit_submit)
    final_forecast_df = prod_orchestrator.fit_predict(clean_data)

    final_forecast_df.to_csv('./submit.csv', index=False)
