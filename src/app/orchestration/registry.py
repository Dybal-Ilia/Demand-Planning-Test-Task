from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..predictors.models import (
    BaseModel,
    HoltWintersForecaster,
    MovingAverageForecaster,
    SESForecaster,
)


@dataclass
class ModelConfig:
    """
    Конфигурация для инициализации модели прогнозирования.
    Хранит ссылку на класс модели и гиперпараметры по умолчанию.
    """
    forecaster_class: type[BaseModel]
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def create_instance(self) -> BaseModel:
        """Фабричный метод: создает новый чистый экземпляр модели с нужными параметрами."""
        return self.forecaster_class(**self.params)


class ForecastingRegistry:
    """
    Реестр правил маршрутизации. Отделяет конфигурацию от логики выполнения.
    """
    _routes: ClassVar[dict[str, ModelConfig]] = {
        'A': ModelConfig(
            forecaster_class=HoltWintersForecaster, 
            params={'seasonal_periods': 12},
            description="Тройное эксп. сглаживание для товаров со стабильной сезонностью"
        ),
        'B': ModelConfig(
            forecaster_class=SESForecaster,
            description="Простое эксп. сглаживание для товаров со средним объемом продаж"
        ),
        'C': ModelConfig(
            forecaster_class=MovingAverageForecaster, 
            params={'window_size': 3},
            description="Скользящее среднее для товаров с редким/нестабильным спросом"
        )
    }

    @classmethod
    def get_model(cls, abc_class: str, default_class: str = 'C') -> BaseModel:
        """
        Возвращает готовый к обучению экземпляр модели для указанного сегмента.
        Если сегмент неизвестен, использует default_class (фолбэк).
        """
        config = cls._routes.get(abc_class)
        if config is None:
            config = cls._routes[default_class]
        return config.create_instance()