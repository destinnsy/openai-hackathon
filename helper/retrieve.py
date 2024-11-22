from .db import QueryResult, query_index

MIN_DISTANCE=1.3
K=10

def _build_context(query_result: QueryResult, min_distance:float)->str:
    documents = query_result.documents[0]
    distances = query_result.distances[0]
    context = ["Here are some notes from previous conversations between you and the user that might be relevant to you. Note that this snippets are from conversations that happened in the past."]
    context_num = 1
    seen_contexts = set() # to handle exact duplicates that could inadvertedly be in the index
    for document, distance in zip(documents, distances):
        if distance < min_distance and document not in seen_contexts:
            context.append(f"{context_num}: {document}")
            context_num += 1
            seen_contexts.add(document)
    return "\n".join(context)


def retrieve(content_to_retrieve:str, min_distance:float=MIN_DISTANCE, k:int=K):
    query_result = query_index(
        query_texts=[content_to_retrieve],
        n_results=k
    )
    return _build_context(query_result, min_distance)
