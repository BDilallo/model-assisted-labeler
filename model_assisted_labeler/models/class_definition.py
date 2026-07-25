from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDefinition:
    """Represents one session annotation class."""

    class_id: int
    name: str

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("Class ID cannot be negative.")

        cleaned_name = self.name.strip()

        if not cleaned_name:
            raise ValueError("Class name cannot be empty.")

        object.__setattr__(self, "name", cleaned_name)
