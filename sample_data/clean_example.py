"""Well-structured data processing utilities.

This module provides clean, well-documented functions for common data
transformation tasks. Designed to demonstrate best practices:
well-named identifiers, type hints, docstrings, and low complexity.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Optional


def read_csv_records(file_path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    """Read a CSV file and return rows as a list of dictionaries.

    Args:
        file_path: Path to the CSV file to read.
        delimiter: Column delimiter character (default: comma).

    Returns:
        List of row dictionaries mapping header names to string values.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        return list(reader)


def filter_records(
    records: list[dict[str, str]],
    field: str,
    min_value: float,
) -> list[dict[str, str]]:
    """Filter records where a numeric field exceeds a minimum threshold.

    Args:
        records: List of record dictionaries to filter.
        field: Key name of the numeric field to compare.
        min_value: Minimum acceptable value (inclusive).

    Returns:
        Filtered list of records where field >= min_value.
    """
    filtered = []
    for record in records:
        raw = record.get(field, "")
        try:
            value = float(raw)
        except (ValueError, TypeError):
            continue
        if value >= min_value:
            filtered.append(record)
    return filtered


def compute_field_statistics(
    records: list[dict[str, str]],
    field: str,
) -> dict[str, float]:
    """Compute descriptive statistics for a numeric field across records.

    Args:
        records: List of record dictionaries.
        field: Key name of the field to analyze.

    Returns:
        Dictionary with keys: mean, median, stdev, min, max.
        Returns empty dict if no valid numeric values are found.
    """
    numeric_values: list[float] = []
    for record in records:
        raw = record.get(field, "")
        try:
            numeric_values.append(float(raw))
        except (ValueError, TypeError):
            continue

    if not numeric_values:
        return {}

    return {
        "mean": statistics.mean(numeric_values),
        "median": statistics.median(numeric_values),
        "stdev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def group_records_by_field(
    records: list[dict[str, str]],
    group_field: str,
) -> dict[str, list[dict[str, str]]]:
    """Group records by the value of a specified field.

    Args:
        records: List of record dictionaries to group.
        group_field: Field name to group by.

    Returns:
        Dictionary mapping field values to lists of matching records.
    """
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        key = record.get(group_field, "")
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    return groups


def write_csv_records(
    records: list[dict[str, str]],
    output_path: Path,
    fieldnames: Optional[list[str]] = None,
) -> int:
    """Write a list of record dictionaries to a CSV file.

    Args:
        records: Records to write (must share the same keys).
        output_path: Destination path for the output CSV file.
        fieldnames: Column order. Inferred from first record if None.

    Returns:
        Number of records written to the file.
    """
    if not records:
        return 0

    if fieldnames is None:
        fieldnames = list(records[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    return len(records)


def summarize_dataset(
    records: list[dict[str, str]],
    numeric_fields: list[str],
) -> dict[str, dict[str, float]]:
    """Compute statistics for multiple numeric fields at once.

    Args:
        records: The dataset as a list of row dictionaries.
        numeric_fields: Names of fields to compute statistics for.

    Returns:
        Nested dict: {field_name: {mean, median, stdev, min, max}}.
    """
    return {
        field: compute_field_statistics(records, field)
        for field in numeric_fields
    }
