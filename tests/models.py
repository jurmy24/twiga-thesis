from pydantic import BaseModel, Field, conint

class HitResponse(BaseModel):
    hit: int = Field(description="Either the number 0 or 1")