# Here is an example search query in elasticsearch
results = dataLoader.search(
    query={
        'match': {
            'name': {
                'query': "some text that I search on to match the name of a document"
            }
        }
    }
)

# If I want to search on multiple fields at the same time using BM25 I can do something like this (here I search on the name, summary, and content)
query={
        'multi_match': {
            'query': "some text that I search on to match the name, summary, or content of a document",
            'fields': ['name', 'summary', 'content'],
        }
    }

# If I want to add a metadata filter of sorts I can add a boolean query to my search
query={
    'bool': {
        'must': [{
            'multi_match': {
                'query': "query text here",
                'fields': ['name', 'summary', 'content'],
            }
        }],
        'filter': [{
            'term': {
                'category.keyword': {
                    'value': "category to filter"
                }
            }
        }]
    }
}


