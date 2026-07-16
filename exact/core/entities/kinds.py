from enum import Enum


class EntityKind(str, Enum):
    """Kinds of named entities exposed by a knowledge source."""

    CLASS = "class"
    OBJECT_PROPERTY = "object_property"
    DATA_PROPERTY = "data_property"
    ANNOTATION_PROPERTY = "annotation_property"
    INDIVIDUAL = "individual"
