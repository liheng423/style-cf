from dataclasses import dataclass

@dataclass(frozen=True)
class Col:
    ID: str = "ID"
    TIME: str = "TIME"
    LANE: str = "LANE"
    SPD: str = "SPD"
    ACC: str = "ACC"
    KILO: str = "KILO"
    LEN: str = "LEN"
    DECT: str = "DECT"
    LC: str = "LC"
    IN_LC: str = "IN_LC"
