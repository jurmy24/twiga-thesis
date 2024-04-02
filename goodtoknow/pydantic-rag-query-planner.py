import enum
from typing import List
from pydantic import Field, BaseModel

import instructor
from openai import OpenAI


class QueryType(str, enum.Enum):
    """Enumeration representing the types of queries that can be asked to a question answer system."""

    SINGLE_QUESTION = "SINGLE"
    MERGE_MULTIPLE_RESPONSES = "MERGE_MULTIPLE_RESPONSES"


class Query(BaseModel):
    """Class representing a single question in a query plan."""

    id: int = Field(..., description="Unique id of the query")
    question: str = Field(
        ...,
        description="Question asked using a question answering system",
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub questions that need to be answered before asking this question",
    )
    node_type: QueryType = Field(
        default=QueryType.SINGLE_QUESTION,
        description="Type of question, either a single question or a multi-question merge",
    )


class QueryPlan(BaseModel):
    """Container class representing a tree of questions to ask a question answering system."""

    query_graph: List[Query] = Field(
        ..., description="The query graph representing the plan"
    )

    def _dependencies(self, ids: List[int]) -> List[Query]:
        """Returns the dependencies of a query given their ids."""
        return [q for q in self.query_graph if q.id in ids]
    


# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())


def query_planner(question: str) -> QueryPlan:
    PLANNING_MODEL = "gpt-4-0613"

    messages = [
        {
            "role": "system",
            "content": "You are a world class query planning algorithm capable ofbreaking apart questions into its dependency queries such that the answers can be used to inform the parent question. Do not answer the questions, simply provide a correct compute graph with good specific questions to ask and relevant dependencies. Before you call the function, think step-by-step to get a better understanding of the problem.",
        },
        {
            "role": "user",
            "content": f"Consider: {question}\nGenerate the correct query plan.",
        },
    ]

    root = client.chat.completions.create(
        model=PLANNING_MODEL,
        temperature=0,
        response_model=QueryPlan,
        messages=messages,
        max_tokens=1000,
    )
    return root

plan = query_planner(
    "What is the difference in populations of Canada and the Jason's home country?"
)
plan.model_dump()

# While we build the query plan in this example, we do not propose a method to actually answer the question. 
# You can implement your own answer function that perhaps makes a retrieval and calls openai for retrieval augmented generation. 
# That step would also make use of function calls but goes beyond the scope of this example.

"""
{
    "query_graph": [
        {
            "dependencies": [],
            "id": 1,
            "node_type": "SINGLE",
            "question": "Identify Jason's home country",
        },
        {
            "dependencies": [],
            "id": 2,
            "node_type": "SINGLE",
            "question": "Find the population of Canada",
        },
        {
            "dependencies": [1],
            "id": 3,
            "node_type": "SINGLE",
            "question": "Find the population of Jason's home country",
        },
        {
            "dependencies": [2, 3],
            "id": 4,
            "node_type": "SINGLE",
            "question": "Calculate the difference in populations between Canada and Jasons home country",
        },
    ]
}
"""