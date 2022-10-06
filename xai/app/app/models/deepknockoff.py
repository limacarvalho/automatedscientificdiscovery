import pandas as pd
from pydantic import BaseModel


class DMDeepKnockoff(BaseModel):
    df: pd.DataFrame
    Sigma: int
    ksampler: str
    fstat: str
    fdr: 0.1

    class Config:
        arbitrary_types_allowed = True

