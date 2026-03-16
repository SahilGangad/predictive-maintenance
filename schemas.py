# schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class MachineInput(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(..., description="Machine quality type")
    air_temperature: float = Field(..., ge=295, le=305, description="Air temp in Kelvin")
    process_temperature: float = Field(..., ge=305, le=315, description="Process temp in Kelvin")
    rotational_speed: float = Field(..., ge=1000, le=3000, description="RPM")
    torque: float = Field(..., ge=0, le=80, description="Torque in Nm")
    tool_wear: float = Field(..., ge=0, le=300, description="Tool wear in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "machine_type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0
            }
        }

class PredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float
    risk_level: str
    message: str