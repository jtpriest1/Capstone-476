from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class ColumnMap(BaseSettings):
    """Maps logical names to actual dataset column names. Override per-request in POST /train."""
    text_column: str = "text"
    label_column: str = "label"
    label_positive_value: str = "spam"  # the value in label_column that means "scam/spam"

    model_config = SettingsConfigDict(env_prefix="DATASET_")


class Settings(BaseSettings):
    model_dir: Path = Path("models")
    dataset_dir: Path = Path("datasets")
    default_model_name: str = "tfidf_logreg"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Default column mapping — can be overridden per training request
    dataset_text_column: str = "text"
    dataset_label_column: str = "label"
    dataset_label_positive_value: str = "spam"


    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


    @property
    def default_column_map(self) -> ColumnMap:
        return ColumnMap(
            text_column=self.dataset_text_column,
            label_column=self.dataset_label_column,
            label_positive_value=self.dataset_label_positive_value,
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()
