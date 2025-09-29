from __future__ import annotations
import os
import re
import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ruamel.yaml import YAML


class SchemaValidationError(Exception):
    """Raised when the CSV rows violate the schema."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        message = "Schema validation failed:\n" + "\n".join(self.errors)
        super().__init__(message)


class SchemaLoadingError(Exception):
    """Raised when a schema file cannot be parsed."""


TRUE_VALUES = {"true", "t", "yes", "y", "1", "on"}
FALSE_VALUES = {"false", "f", "no", "n", "0", "off"}
NULL_LITERALS = {"", "null", "none", "NULL", "None", None}


@dataclass
class FieldSpec:
    key: str
    type: str
    required: bool = False
    default: Any = None
    allow_null: bool = False
    delimiter: str = ","
    choices: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    regex: Optional[str] = None
    path_exists: bool = False
    allow_relative: bool = True
    description: Optional[str] = None

    def __post_init__(self) -> None:
        pattern = self.regex
        self._compiled_regex = re.compile(pattern) if pattern else None

    @property
    def compiled_regex(self):
        return self._compiled_regex


@dataclass
class RowValidationResult:
    exp_id: str
    values_to_set: Dict[str, Any]
    final_values: Dict[str, Any]


class Schema:
    def __init__(
        self,
        name: str,
        strict_columns: bool,
        allow_missing_optionals: bool,
        defaults: Dict[str, Any],
        fields: List[FieldSpec],
        rules: List[Dict[str, Any]],
    ) -> None:
        self.name = name
        self.strict_columns = strict_columns
        self.allow_missing_optionals = allow_missing_optionals
        self.defaults = defaults or {}
        self.fields = {field.key: field for field in fields}
        self.rules = rules or []

    @classmethod
    def from_yaml(cls, path: str) -> "Schema":
        yaml = YAML()
        try:
            with open(path, "r") as f:
                data = yaml.load(f)
        except Exception as exc:  # pragma: no cover - read failure already clear
            raise SchemaLoadingError(f"Failed to load schema '{path}': {exc}") from exc

        if data is None:
            raise SchemaLoadingError(f"Schema file '{path}' is empty")

        fields_data = data.get("fields") or []
        fields: List[FieldSpec] = []
        for field_dict in fields_data:
            field_kwargs = dict(field_dict)
            if "min" in field_kwargs and field_kwargs["min"] is None:
                field_kwargs.pop("min")
            if "max" in field_kwargs and field_kwargs["max"] is None:
                field_kwargs.pop("max")
            fields.append(FieldSpec(**field_kwargs))

        return cls(
            name=data.get("schema", os.path.basename(path)),
            strict_columns=data.get("strict_columns", False),
            allow_missing_optionals=data.get("allow_missing_optionals", True),
            defaults=data.get("defaults", {}),
            fields=fields,
            rules=data.get("rules", []),
        )

    @property
    def field_keys(self) -> List[str]:
        return list(self.fields.keys())

    def validate_columns(self, csv_columns: List[str]) -> None:
        csv_columns = csv_columns or []
        csv_column_set = set(csv_columns)

        errors: List[str] = []

        for field in self.fields.values():
            if (field.required and field.default is None and field.key not in csv_column_set):
                errors.append(f"Missing required column '{field.key}' in CSV header")

        if self.strict_columns:
            unknown_columns = csv_column_set - set(self.fields.keys())
            for column in sorted(unknown_columns):
                errors.append(f"Column '{column}' is not defined in schema '{self.name}'")

        if errors:
            raise SchemaValidationError(errors)

    def validate_rows(self, rows: List[Dict[str, Any]]) -> List[RowValidationResult]:
        errors: List[str] = []
        results: List[RowValidationResult] = []

        for idx, row in enumerate(rows):
            row_result, row_errors = self._validate_row(row, idx)
            if row_errors:
                errors.extend(row_errors)
            else:
                results.append(row_result)

        if errors:
            raise SchemaValidationError(errors)
        return results

    # ------------------------------------------------------------------
    # Row-level helpers
    # ------------------------------------------------------------------
    def _validate_row(
        self, row: Dict[str, Any], row_index: int
    ) -> Tuple[Optional[RowValidationResult], List[str]]:
        errors: List[str] = []
        values_to_set: Dict[str, Any] = {}
        final_values: Dict[str, Any] = {}

        for key, field in self.fields.items():
            raw_present = key in row and row[key] is not None
            raw_value = row.get(key)
            raw_str = raw_value.strip() if isinstance(raw_value, str) else raw_value

            value_set = False
            final_value: Any = None

            if not raw_present:
                if field.default is not None:
                    final_value = field.default
                    value_set = True
                elif field.required:
                    errors.append(
                        self._format_error(row_index, key, "missing required value")
                    )
                elif not self.allow_missing_optionals:
                    final_value = None
                    value_set = True
                else:
                    final_value = None
                    value_set = False
            else:
                if self._is_null_literal(raw_str):
                    if field.allow_null:
                        final_value = None
                        value_set = True
                    elif field.required:
                        errors.append(
                            self._format_error(row_index, key, "value cannot be null")
                        )
                    else:
                        final_value = None
                        value_set = False
                else:
                    try:
                        final_value = self._convert_value(field, raw_str)
                        value_set = True
                    except ValueError as exc:
                        errors.append(
                            self._format_error(row_index, key, str(exc))
                        )
                        final_value = None
                        value_set = False

            final_values[key] = final_value
            if value_set:
                values_to_set[key] = final_value

        if not errors:
            rule_errors = self._apply_rules(final_values, row_index)
            if rule_errors:
                errors.extend(rule_errors)

        if errors:
            return None, errors

        exp_value = final_values.get("exp_id")
        if not isinstance(exp_value, str) or not exp_value:
            return None, [self._format_error(row_index, "exp_id", "invalid experiment id")]

        return RowValidationResult(
            exp_id=exp_value,
            values_to_set=values_to_set,
            final_values=final_values,
        ), []

    @staticmethod
    def _format_error(row_index: int, key: str, message: str) -> str:
        # +2 accounts for header row in CSV (1-based indexing)
        return f"Row {row_index + 2}, column '{key}': {message}"

    @staticmethod
    def _is_null_literal(value: Any) -> bool:
        if value in NULL_LITERALS:
            return True
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return True
        return False

    def _convert_value(self, field: FieldSpec, raw_value: Any) -> Any:
        value: Any = raw_value

        if field.type == "str":
            value = str(raw_value)
        elif field.type == "int":
            try:
                value = int(str(raw_value))
            except ValueError as exc:
                raise ValueError("expected integer") from exc
        elif field.type == "float":
            try:
                value = float(str(raw_value))
            except ValueError as exc:
                raise ValueError("expected float") from exc
        elif field.type == "bool":
            value = self._convert_bool(raw_value)
        elif field.type == "path":
            value = self._convert_path(raw_value, field)
        elif field.type.startswith("list["):
            value = self._convert_list(raw_value, field)
        elif field.type == "json":
            try:
                value = ast.literal_eval(str(raw_value))
            except (ValueError, SyntaxError) as exc:
                raise ValueError("expected valid Python literal") from exc
        else:
            raise ValueError(f"unsupported field type '{field.type}'")

        if field.choices is not None:
            if isinstance(value, list):
                invalid = [item for item in value if item not in field.choices]
                if invalid:
                    raise ValueError(
                        f"value(s) {invalid} not in allowed choices {field.choices}"
                    )
            elif value not in field.choices:
                raise ValueError(
                    f"value '{value}' not in allowed choices {field.choices}"
                )

        if field.min is not None:
            if not isinstance(value, (int, float)):
                raise ValueError("minimum constraint requires numeric value")
            if value < field.min:
                raise ValueError(f"value {value} below minimum {field.min}")

        if field.max is not None:
            if not isinstance(value, (int, float)):
                raise ValueError("maximum constraint requires numeric value")
            if value > field.max:
                raise ValueError(f"value {value} above maximum {field.max}")

        if field.compiled_regex is not None:
            if not isinstance(value, str) or not field.compiled_regex.match(value):
                raise ValueError("value does not match required pattern")

        return value

    def _convert_bool(self, raw_value: Any) -> bool:
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return bool(raw_value)
        value_str = str(raw_value).strip().lower()
        if value_str in TRUE_VALUES:
            return True
        if value_str in FALSE_VALUES:
            return False
        raise ValueError("expected boolean")

    def _convert_path(self, raw_value: Any, field: FieldSpec) -> str:
        path_value = str(raw_value).strip()
        if not os.path.isabs(path_value) and not field.allow_relative:
            raise ValueError("expected absolute path")
        if field.path_exists and not os.path.exists(path_value):
            raise ValueError(f"path does not exist: {path_value}")
        return path_value

    def _convert_list(self, raw_value: Any, field: FieldSpec) -> List[Any]:
        if isinstance(raw_value, list):
            items = raw_value
        else:
            items = [item.strip() for item in str(raw_value).split(field.delimiter)]
        items = [item for item in items if item != ""]
        subtype = field.type[len("list[") : -1]
        if subtype == "str":
            return items
        if subtype == "int":
            converted: List[int] = []
            for item in items:
                try:
                    converted.append(int(item))
                except ValueError as exc:
                    raise ValueError("expected integer in list") from exc
            return converted
        if subtype == "float":
            converted_float: List[float] = []
            for item in items:
                try:
                    converted_float.append(float(item))
                except ValueError as exc:
                    raise ValueError("expected float in list") from exc
            return converted_float
        raise ValueError(f"unsupported list subtype '{subtype}'")

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------
    def _apply_rules(
        self,
        final_values: Dict[str, Any],
        row_index: int,
    ) -> List[str]:
        errors: List[str] = []
        for rule in self.rules:
            rule_type = rule.get("type")
            if not rule_type:
                continue
            if rule_type == "require_if":
                if self._condition_matches(rule.get("if", {}), final_values):
                    errors.extend(
                        self._evaluate_require_if(rule, final_values, row_index)
                    )
            elif rule_type == "forbid_if":
                if self._condition_matches(rule.get("if", {}), final_values):
                    errors.extend(
                        self._evaluate_forbid_if(rule, final_values, row_index)
                    )
            elif rule_type == "range_if":
                errors.extend(
                    self._evaluate_range_if(rule, final_values, row_index)
                )
        return errors

    def _condition_matches(self, condition: Dict[str, Any], values: Dict[str, Any]) -> bool:
        if not condition:
            return True
        key = condition.get("key")
        if key is None:
            return True
        value = values.get(key)

        if "equals" in condition and value != condition["equals"]:
            return False
        if "not_equals" in condition and value == condition["not_equals"]:
            return False
        if "in" in condition and value not in condition["in"]:
            return False
        if "not_in" in condition and value in condition["not_in"]:
            return False
        if "is_null" in condition:
            is_null = condition["is_null"]
            if is_null and value is not None:
                return False
            if not is_null and value is None:
                return False
        return True

    def _evaluate_require_if(
        self,
        rule: Dict[str, Any],
        values: Dict[str, Any],
        row_index: int,
    ) -> List[str]:
        errors: List[str] = []
        required_keys = rule.get("required_keys", [])
        for key in required_keys:
            if values.get(key) in (None, ""):
                errors.append(self._format_error(row_index, key, "value is required"))

        required_values = rule.get("required_values", {})
        for key, expected in required_values.items():
            if values.get(key) != expected:
                errors.append(
                    self._format_error(
                        row_index,
                        key,
                        f"expected value '{expected}' when condition holds",
                    )
                )
        return errors

    def _evaluate_forbid_if(
        self,
        rule: Dict[str, Any],
        values: Dict[str, Any],
        row_index: int,
    ) -> List[str]:
        errors: List[str] = []
        forbidden_keys = rule.get("forbidden_when_set", [])
        for key in forbidden_keys:
            if values.get(key) not in (None, "", -1):
                errors.append(
                    self._format_error(
                        row_index,
                        key,
                        "value is not allowed when condition holds",
                    )
                )
        return errors

    def _evaluate_range_if(
        self,
        rule: Dict[str, Any],
        values: Dict[str, Any],
        row_index: int,
    ) -> List[str]:
        condition = rule.get("if", {})
        if not self._condition_matches(condition, values):
            return []
        key = rule.get("key")
        if not key:
            return []
        value = values.get(key)
        errors: List[str] = []
        if value is None:
            errors.append(self._format_error(row_index, key, "value is required"))
            return errors
        minimum = rule.get("min")
        maximum = rule.get("max")
        if minimum is not None and value < minimum:
            errors.append(
                self._format_error(
                    row_index,
                    key,
                    f"value {value} below minimum {minimum}",
                )
            )
        if maximum is not None and value > maximum:
            errors.append(
                self._format_error(
                    row_index,
                    key,
                    f"value {value} above maximum {maximum}",
                )
            )
        return errors


def load_schema(schema_path: str) -> Schema:
    return Schema.from_yaml(schema_path)
