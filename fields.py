"""Utilities for working with observing-field metadata."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple


class FieldCatalog:
    """Normalized view of coadd files organized by observing field."""

    def __init__(self, mapping: Dict[str, Iterable[str]] | None):
        normalized: Dict[str, Tuple[str, ...]] = {}
        if mapping is None:
            mapping = {}

        for field, paths in mapping.items():
            if field is None:
                raise ValueError("Field names must be non-null strings.")
            field_name = str(field)
            if isinstance(paths, (str, bytes)):
                path_list = [paths]
            else:
                path_list = list(paths)
            if not path_list:
                raise ValueError(f"Field '{field_name}' is configured with no coadd files.")
            normalized[field_name] = tuple(str(path) for path in path_list)

        self._field_to_paths: Dict[str, Tuple[str, ...]] = dict(sorted(normalized.items()))
        path_map: Dict[str, List[str]] = {}
        for field_name, paths in self._field_to_paths.items():
            for path in paths:
                path_map.setdefault(path, []).append(field_name)
        self._path_to_fields: Dict[str, Tuple[str, ...]] = {path: tuple(sorted(fields)) for path, fields in path_map.items()}

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Return field names sorted alphabetically."""
        return tuple(self._field_to_paths.keys())

    @property
    def all_paths(self) -> Tuple[str, ...]:
        """Return a flattened tuple of all coadd file paths."""
        return tuple(path for _, path in self.iter_field_paths())

    def iter_field_paths(self) -> Iterator[Tuple[str, str]]:
        """Yield (field, path) pairs for each configured coadd file."""
        for field, paths in self._field_to_paths.items():
            for path in paths:
                yield field, path

    def paths_for_field(self, field: str) -> Tuple[str, ...]:
        """Return all coadd paths associated with a given field."""
        return self._field_to_paths[str(field)]

    def fields_for_path(self, path: str) -> Tuple[str, ...]:
        """Return all field names associated with a particular coadd path."""
        return self._path_to_fields.get(str(path), tuple())

    def as_serializable(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return a hashable representation for caching."""
        return tuple((field, tuple(paths)) for field, paths in self._field_to_paths.items())
