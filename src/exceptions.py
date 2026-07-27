class ModelNotLoadedError(Exception):
    """Raised when model cannot be loaded."""


class InvalidImageError(Exception):
    """Raised when uploaded image is invalid."""


class PredictionError(Exception):
    """Raised during inference."""


class ConfigurationError(Exception):
    """Raised when config is invalid."""