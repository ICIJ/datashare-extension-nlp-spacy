from enum import Enum, unique

from icij_common.pydantic_utils import LowerCamelCaseModel


@unique
class Category(str, Enum):
    PER = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    NUM = "NUMBER"
    UNK = "UNKNOWN"


class NamedEntity(LowerCamelCaseModel):
    start: int
    text: str
    category: Category


@unique
class SpacySize(str, Enum):
    SMALL = "sm"
    MEDIUM = "md"
    LARGE = "lg"
    TRANSFORMER = "trf"
