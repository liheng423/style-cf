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

@dataclass(frozen=True)
class HH_Col:
    ID: str = "case_id"
    TIME: str = "time"
    LANE: str = None
    SPD: str = "SPD"
    ACC: str = "ACC"
    KILO: str = "KILO"
    LEN: str = "LEN"
    DECT: str = "DECT"
    LC: str = "LC"

    # Car-Following specific columns
    X_LEAD: str = "x_leader"
    X_FOL: str = "x_follower"
    V_LEAD: str = "v_leader"
    V_FOL: str = "v_follower"
    A_LEAD: str = "a_leader"
    A_FOL: str = "a_follower"
    L_LEAD: str = "l_leader"
    L_FOL: str = "l_follower"