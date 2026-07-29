import json
import os
from pathlib import Path


class ApplicationSettingsRepository:
    """Persist the application's selected save-folder location."""

    SETTINGS_FILENAME = "settings.json"
    WORKSPACE_KEY = "workspace_root"

    def __init__(self, settings_path: Path | None = None) -> None:
        if settings_path is None:
            app_data = os.environ.get("APPDATA")

            if app_data:
                settings_directory = Path(app_data)
            else:
                settings_directory = Path.home() / ".config"

            settings_path = (
                settings_directory
                / "Model-Assisted Labeler"
                / self.SETTINGS_FILENAME
            )

        self._settings_path = Path(settings_path)

    @property
    def settings_path(self) -> Path:
        return self._settings_path

    @staticmethod
    def default_workspace_root() -> Path:
        documents_directory = Path.home() / "Documents"

        if not documents_directory.exists():
            documents_directory = Path.home()

        return documents_directory / "Model-Assisted Labeler"

    def get_workspace_root(self) -> Path | None:
        if not self._settings_path.is_file():
            return None

        try:
            data = json.loads(
                self._settings_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None

        workspace_value = data.get(self.WORKSPACE_KEY)

        if not isinstance(workspace_value, str):
            return None

        workspace_value = workspace_value.strip()

        if not workspace_value:
            return None

        workspace_root = Path(workspace_value)

        if not workspace_root.is_dir():
            return None

        return workspace_root

    def set_workspace_root(self, workspace_root: Path) -> None:
        workspace_root = Path(workspace_root)
        workspace_root.mkdir(parents=True, exist_ok=True)

        self._settings_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self._settings_path.with_suffix(".tmp")
        payload = {
            self.WORKSPACE_KEY: str(workspace_root.resolve()),
        }

        temporary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self._settings_path)
