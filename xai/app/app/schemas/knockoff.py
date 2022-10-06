from typing import Optional, List
from pydantic import BaseModel, Field
import numpy as np
from pydantic.dataclasses import dataclass




# Shared properties
class RejectionBase(BaseModel):
    fdr: Optional[int] = None
    rejections: Optional[list] = None

class KnockoffRejection(RejectionBase):
    pass

class KnockofffilterParam(BaseModel):
    fdr: Optional[int] = 0.1
    sigma: Optional[float] = None
    ksampler: Optional[str] = 'gaussian'
    fstat: Optional[str] = 'lasso'


class DeepknockoffParam(BaseModel):
    epochs: Optional[int] = 100
    epoch_length: Optional[int] = 1000
    family: Optional[str] = 'continuous'
    p: Optional[int] = 0
