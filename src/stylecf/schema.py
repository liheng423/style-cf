from dataclasses import dataclass

@dataclass(frozen=True)
class TensorNames:
    T: str = "T"
    F: str = "F"
    N: str = "N"
    B: str = "B"
    INPUTS: str = "inputs"