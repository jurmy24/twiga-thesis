from pydantic import BaseModel, Field

class HitResponse(BaseModel):
    hit: int = Field(description="Either the number 0 or 1")