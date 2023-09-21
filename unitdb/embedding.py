from typing import Any, List, Literal

import openai

EmbeddingModel = Literal["text-embedding-ada-002"]


def call_openai(
    to_embed: List[str], model: EmbeddingModel = "text-embedding-ada-002", *args: Any, **kwargs: Any
) -> List[List[float]]:
    """
    This function calls the OpenAI API to generate embeddings for a list of strings.

    Args:
        to_embed (List[str]): The list of strings to generate embeddings for.
        model (OpenAIEmbeddingModel, optional): The model to use for embeddings. Defaults to "text-embedding-ada-002".
        *args (Any): Additional arguments to pass to the OpenAI API.
        **kwargs (Any): Additional keyword arguments to pass to the OpenAI API.

    Returns:
        List[List[float]]: A list of embeddings, each of which is a list of floats.
    """
    resp = openai.Embedding.create(input=to_embed, model=model, *args, **kwargs)  # type: ignore
    return [d["embedding"] for d in resp.data]
