from src.models import ChatMessage

"""
From LlamaIndex (llama_index/llama-index-core/llama_index/core/prompts/default_prompts.py)
"""
DEFAULT_TEXT_QA_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_TREE_SUMMARIZE = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_PROMPT = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided context information below. "
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}"
)

# Here is an example using two-shot prompting (i.e. providing examples)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)


"""
Directly from LlamaIndex (llama_index/llama-index-core/llama_index/core/prompts/chat_prompts.py)
"""

# TODO: determine whether to use ChatMessage here or to just use the prompt literal like what is done above

# text qa prompt
CHAT_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines."
    ),
    role="system",
)

CHAT_TEXT_QA_USER_PROMPT = ChatMessage(
    content=(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    ),
    role="user"
)

# Tree Summarize
CHAT_TREE_SUMMARIZE_USER_PROMPT = ChatMessage(
    content=(
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    ),
    role="user",
    )

# Refine Prompt (a variant of this one can be used for the query rewriter)
REFINE_USER_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that strictly operates in two modes "
        "when refining existing answers:\n"
        "1. **Rewrite** an original answer using the new context.\n"
        "2. **Repeat** the original answer if the new context isn't useful.\n"
        "Never reference the original answer or context directly in your answer.\n"
        "When in doubt, just repeat the original answer.\n"
        "New Context: {context_msg}\n"
        "Query: {query_str}\n"
        "Original Answer: {existing_answer}\n"
        "New Answer: "
    ),
    role="user",
    )

REWRITE_QUERY_PROMPT = ChatMessage(
    content=(
        "You are an expert assistant that simply rewrites a query into a short passage about the topic it is requesting a question about. You do not write a question, but only find the topic they are requesting a question about and describe that topic."
    ),
    role="system"
)