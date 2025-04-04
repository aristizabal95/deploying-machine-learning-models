from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""
    print(input_data.columns)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=input_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return input_data, errors

class TitanicDataInputSchema(BaseModel):
    pclass: int
    survived: int
    name: str
    sex: str
    age: float
    sibsp: int
    parch: int
    ticket: int
    fare: float
    cabin: str
    embarked: str
    boat: int
    body: int
    __annotations__["home.dest"] = str


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]