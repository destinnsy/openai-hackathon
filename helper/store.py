import time
from typing import Optional
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field

from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Type, Union, Any
from llama_index.core.output_parsers.utils import parse_json_markdown
import json
from .db import openai_client, insert_to_index, collection


def make_request(model: str, messages: list[dict[str, str]]) -> str:
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
)
def chat_completion_request(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    response_model: Type[BaseModel] = None,
) -> Union[str, dict[str, Any]]:
    try:
        content = make_request(model, messages)
        if response_model is not None:
            parsed_content = parse_json_markdown(content)
            try:
                return response_model(**parsed_content)
            except TypeError as e:
                error_message = {
                    "role": "user",
                    "content": f"JSON decoding error: {e}. Please adhere to the json response format that obeys the following schema: {response_model.model_json_schema()}",
                }
                messages.append(error_message)
                logger.error(
                    f"TypeError in response_model parsing: {e}. Content: {parsed_content}"
                )
                raise
        else:
            return content
    except json.JSONDecodeError as e:
        error_message = {
            "role": "user",
            "content": f"JSON decoding error: {e}. Please adhere to the json response format that obeys the following schema: {response_model.model_json_schema()}",
        }
        messages.append(error_message)
        logger.error(f"JSON decoding error: {e}. Content: {content}")
        raise
    except Exception as e:
        logger.error(f"Error while making chat completion request: {e}")
        raise

class Snippet(BaseModel):
    text: str
    date_of_event: Optional[str] = Field(description="to be filled in if the snippet is an event", default=None) # TODO not sure what to do with this info for now

class ConversationSnippets(BaseModel):
    snippets: list[Snippet]

extract_snippets_from_conversation_prompt = """\
"""

def _get_date_today():
    return datetime.now().strftime("%Y-%m-%d")


def _construct_conversation(user_messages:list[str], assistant_messages:list[str])->str:
    conversation = []
    for user_message, assistant_message in zip(user_messages, assistant_messages):
        conversation.append(f"User: {user_message}")
        conversation.append(f"Confidante: {assistant_message}")
    return "\n".join(conversation)

def extract_snippets_from_conversation(user_messages:list[str], assistant_messages:list[str]):
    conversation = _construct_conversation(user_messages, assistant_messages)
    prompt = extract_snippets_from_conversation_prompt.format(date=_get_date_today(), conversation=conversation)
    conversation_snippets: ConversationSnippets = chat_completion_request(
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_model=ConversationSnippets
    )
    logger.info(f"Successfully extracted {len(conversation_snippets.snippets)} snippets from conversation")
    return conversation_snippets

# insert to index
def insert_snippets_to_index(collection, conversation_snippets: ConversationSnippets):
    insert_to_index(
        collection=collection,
        documents=[snippet.text for snippet in conversation_snippets.snippets],
        metadatas=[{"date_of_event": snippet.date_of_event} if snippet.date_of_event else {"date_of_evebt": ""} for snippet in conversation_snippets.snippets]
    )

def store(user_messages:list[str], assistant_messages:list[str]):
    conversation_snippets = extract_snippets_from_conversation(
        user_messages=user_messages,
        assistant_messages=assistant_messages
    )
    # determine_snippets_to_add_or_delete()
    # delete_documents_from_index()
    print(conversation_snippets)
    insert_snippets_to_index(collection=collection, conversation_snippets=conversation_snippets)
    logger.info(f"There are now {collection.count()} documents in the index")
