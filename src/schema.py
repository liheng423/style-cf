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
    LANE: str = "lane"
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

@dataclass(frozen=True)
class CFNAMES:
    SELF_X = "SELF_X"
    SELF_V = "SELF_V"
    SELF_A = "SELF_A"
    SELF_L = "SELF_L"
    LEAD_X = "LEAD_X"
    LEAD_V = "LEAD_V"
    LEAD_A = "LEAD_A"
    LEAD_L = "LEAD_L"
    DELTA_X = "DELTA_X"
    DELTA_V = "DELTA_V"
    SELF_ID = "SELF_ID"
    LEAD_ID = "LEAD_ID"
    TIME = "TIME"
    REACT = "reaction"
    THW = "timeheadway"
    LC = "LC"

@dataclass(frozen=True)
class FEATNAMES:
    INPUTS = "inputs"