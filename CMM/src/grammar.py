from enum import Enum
from dataclasses import dataclass, field


class StateType(Enum):
    LITERAL = "LITERAL"
    ENUM = "ENUM"
    STRING = "STRING"
    NUMBER = "NUMBER"
    ACCEPT = "ACCEPT"


@dataclass
class GrammarState:
    type: StateType
    literal: str = ""
    literal_pos: int = 0
    enum_options: list[str] = field(default_factory=list)
    enum_pos: int = 0
    in_escape: bool = False
    number_has_minus: bool = False
    number_has_dot: bool = False
    number_has_digits: bool = False


@dataclass
class FunctionState:
    state: GrammarState
    function_name: str = ""
    param_index: int = 0
    token_list: list[int] = field(default_factory=list)
