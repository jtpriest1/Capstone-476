"""Model registry — maps model names to detector instances without coupling the API to any class."""
from pathlib import Path
from app.ml.base import BaseDetector

_REGISTRY: dict[str, BaseDetector] = {}


class ModelNotFoundError(Exception):
    pass


class ModelNotLoadedError(Exception):
    pass


def register(cls: type[BaseDetector]) -> type[BaseDetector]:
    """Class decorator. Usage: @register above any BaseDetector subclass."""
    _REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> BaseDetector:
    """Retrieve a registered detector. Raises ModelNotFoundError if unknown."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ModelNotFoundError(
            f"Model '{name}' is not registered. Available: {available}"
        )
    return _REGISTRY[name]


def get_loaded(name: str) -> BaseDetector:
    """Like get(), but also raises ModelNotLoadedError if not yet trained."""
    detector = get(name)
    if not detector.is_loaded:
        raise ModelNotLoadedError(
            f"Model '{name}' is registered but not loaded. Train it first via POST /train."
        )
    return detector


def list_models() -> list[dict]:
    return [
        {"name": name, "loaded": d.is_loaded, "modality": d.modality}
        for name, d in _REGISTRY.items()
    ]


def try_load_from_disk(model_dir: Path) -> None:
    """On startup, load any persisted model files found in model_dir."""
    for name, detector in _REGISTRY.items():
        model_path = model_dir / f"{name}.joblib"
        if model_path.exists():
            try:
                loaded = detector.__class__.load(model_path)
                _REGISTRY[name] = loaded
                print(f"[registry] Loaded '{name}' from {model_path}")
            except Exception as e:
                print(f"[registry] Failed to load '{name}': {e}")
