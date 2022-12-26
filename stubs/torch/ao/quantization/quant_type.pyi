import enum


class QuantType(enum.IntEnum):
    DYNAMIC: int
    STATIC: int
    QAT: int
    WEIGHT_ONLY: int

def quant_type_to_str(quant_type): ...
