import time
from datetime import datetime
from loguru import logger

from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Type, Union, Any
from llama_index.core.output_parsers.utils import parse_json_markdown
import json
from .db import openai_client, insert_to_index, collection, delete_from_index, update_index

from pydantic import BaseModel, Field
from typing import Optional


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
You are to extract snippets of a given conversation between a career confidante and a user, which the confidante should take node of. Think of it as the confidante jotting key points down during the conversation in their journal.
Each snippet has to contain sufficient information to stand alone and be understood without the context of the entire conversation.

**
IMPORTANT: Only return the output in JSON format. The JSON structure should be a list of snippet objects, each with the fields:
	•	"text" (str): The extracted text snippet from the conversation.
	•	"date_of_event" (string): The date of the event mentioned in the snippet. If the snippet is not about an event, this field should be null. Date shouuld be formatted as "YYYY-MM-DD".

Example conversation that happend on 2024-02-01:
User: I am a software engineer and I am considering a career change.
Confidante: What are you considering?
User: I am considering becoming a data scientist.
Confidante: What is motivating you to make this change?
User: I am interested in working with data and I want to leverage my programming skills. I am also going to start taking a course in data science.
Confidante: That's awesome, when do you plan to start the course?
User: I plan to start next month.
Confidante: Great!

Example JSON:
{{
    "snippets": [
        {{
            "text": "User is considering becoming a data scientist.",
            "date_of_event": null
        }},
        {{
            "text": "User is interested in working with data and wants to leverage programming skills. User is also going to start taking a course in data science.",
            "date_of_event": null
        }},
        {{
            "text": "User plans to start data science course next month.",
            "date_of_event": "2024-03-01"
        }}

    ]
}}
===== END OF EXAMPLE ======

The 'snippets' key must be a list of snippets.
The result must be a list of objects with 'text' and 'date_of_event' keys.
Ensure each snippet contains sufficient information to stand alone and be understood without the context of the entire conversation.
**

Conversation that happened on {date}:
{conversation}

JSON:
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



class CurrentKnowledge(BaseModel):
    knowledge: Optional[str] = Field(description="The current knowledge of the user", default=None)

current_knowledge = CurrentKnowledge(knowledge='User mentioned that they just got laid off from a business analyst role, and is actively taking data science courses in order to upskill themselves. The layoff was unexpected and impacts them badly, as the user is the sole breadwinner for a family of 4. They are middle-income, but with two young kids, expenses are a bit tight. They mentioned it would be ideal to have a new job in the next half year if possible, and is willing to upskill themselves or pivot industries/roles.')

UPDATE_KNOWLEDGE_PROMPT = """\
You are a career confidante. Given a conversation that just happened between you and the user, and your current knowledge of the user, update your knowledge of the user.
In your updated knowledge, you should include useful information for future interactions with the user.
The conversation just happened, so you should integrate the new information from the conversation into your updated knowledge. 

Return only the updated knowledge in JSON format.

The conversation is as follows:
{conversation}

Your current knowledge of the user is as follows:
{knowledge}

Response Format:
{{
    "knowledge": "Updated knowledge of the user."
}}
"""

def update_knowledge(user_messages:list[str], assistant_messages:list[str], input_knowledge: CurrentKnowledge):
    conversation = _construct_conversation(user_messages, assistant_messages)
    prompt = UPDATE_KNOWLEDGE_PROMPT.format(conversation=conversation, knowledge=input_knowledge)
    global current_knowledge
    current_knowledge = chat_completion_request(
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_model=CurrentKnowledge
    )
    print(current_knowledge)
    return current_knowledge



from typing import Optional
from datetime import datetime


class TaggedSnippetsWithDbActions(BaseModel):
    snippets_to_add: Optional[list[Snippet]] = Field(
        description="The snippets to add to the database. Do not need id for these, as their ids will be generated upon insertion into the databse.",
        default=None)
    snippets_to_update: Optional[list[Snippet]] = Field(description="The snippets to update in the database",
                                                        default=None)
    snippets_to_delete: Optional[list[Snippet]] = Field(description="The snippets to delete from the database",
                                                        default=None)


TAG_PROMPT = """\
TASK
You can imagine that you are maintaining a journal of the user's career journey. 
Your task is to decide which snippets to add, update and delete in order to maintain a coherent memory the user.
You should return ids and texts of snippets to add to the database.
You are allowed to modify the text to maintain a coherent memory, but ensure the ids remain the same.
You will be shown latest conversation snippets and prior snippets that are related to the current conversation.
You are careful to insert the latest snippets while updating/deleting prior related snippets in order to maintain a coherent memory of the user's career journey.

Prior related snippets sare extracted from an existing database(journal), and should either be deleted or updated based on the latest conversation text snippets. Ensure that the ids match the ids of the snippets in the database.
Latest conversation texts are from the latest conversation between the user and their career confidant and should either be ignored or added. There is NO NEED to add the ids for them.

You are provided with snippets of the latest conversation between a user and their career confidant, and prior related snippets that are already in memory.

**
EXAMPLE_INPUT:
{{
    "latest_conversation_snippet_texts from 2024-11-20": [
        "User previously considred becoming a data scientist.",
        "User is considering becoming a softare engineer.",
        "User has tried the Data Science course, and it doesn't really interest them."
        "User got laid off from their job.",
    ]
    "prior_related_snippets_extracted_from_db": [
        {{
            "text": "User is considering becoming a data scientist.",
            "id": '644ab910-aac1-45c8-acc0-1eef35d9f4e3'
        }},
        {{
            "text": "User is interested in working with data and wants to leverage programming skills. User is also going to start taking a course in data science.",
            "id": '49fbf3e3-3e68-4b0d-9df1-747af9778e94'
        }},
        {{
            "text": User just got laid off from their job, yesterday",
            "date_of_event": "2024-11-18"
            "id": 'ce617a99-cc06-478c-9ddb-7f041572a139',
        }},
    ]
}}

EXAMPLE_OUTPUT:
{{
    "snippets_to_add": [
        {{
            "text": "User is considering becoming a softare engineer.",
        }},
    ],
    "snippets_to_update": [
        {{
            "text": "User has tried the Data Science course, and it doesn't really interest them.",
            "id": '49fbf3e3-3e68-4b0d-9df1-747af9778e94'
        }}
    ],
    "snippets_to_delete": []
}}

**

OUTPUT FORMAT:
{output_format}

INPUT:
{input}
"""


def _format_input(conversation_snippets: ConversationSnippets, prior_related_snippets: list[Snippet]) -> str:
    print(conversation_snippets)
    latest_conversation_snippet_texts = [snippet.text for snippet in conversation_snippets.snippets]
    # prior_related_snippets_extracted_from_db = [{"text": snippet.text, "id": snippet.id, "date_of_event": snippet.date_of_event} for snippet in prior_related_snippets.snippets]
    return str({
        "latest_conversation_snippet_texts": latest_conversation_snippet_texts,
        "prior_related_snippets_extracted_from_db": [obj.model_dump() for obj in prior_related_snippets]
    })


def _tag_db_action_to_snippet(conversation_snippets: ConversationSnippets, prior_related_snippets: list[Snippet],
                              model: str) -> TaggedSnippetsWithDbActions:
    prompt = TAG_PROMPT.format(
        output_format=TaggedSnippetsWithDbActions.model_json_schema(),
        input=_format_input(conversation_snippets, prior_related_snippets)
    )
    tag_snippets_with_db_actions: TaggedSnippetsWithDbActions = chat_completion_request(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_model=TaggedSnippetsWithDbActions
    )
    logger.info(f"Successfully tagged snippets with db actions")
    return tag_snippets_with_db_actions

# tag_snippets_with_db_actions = _tag_db_action_to_snippet(
#     conversation_snippets=ConversationSnippets(
#         snippets=[Snippet(text="User wants to be a software_engineer.", date_of_event="2024-11-20")],
#     ),
#     prior_related_snippets=test_snippets.snippets,
#     model="o1-preview",
# )

from .db import QueryResult, query_index

def _retrieve_related_snippets(snippet: Snippet, n_results: int = 3) -> QueryResult:
    print(snippet)
    query_result = query_index(
        query_texts=[snippet.text],
        n_results=n_results,
    )
    return query_result

def _process_related_snippets(query_result: QueryResult)->list[Snippet]:
    related_snippets = []
    for doc, id, metadata in zip(query_result.documents[0], query_result.ids[0], query_result.metadatas[0]):
        if metadata is not None and metadata['date_of_event'] is not None:
            related_snippets.append(Snippet(text=doc, date_of_event=metadata['date_of_event'], id=id))
        else:
            related_snippets.append(Snippet(text=doc, date_of_event='', id=id,))
    return related_snippets

def _clean_tagged_snippets_with_db_actions(prev_ids, tagged_snippets_with_db_actions: TaggedSnippetsWithDbActions)->tuple[list[Snippet], list[Snippet], list[Snippet]]:
    snippets_to_add = tagged_snippets_with_db_actions.snippets_to_add
    snippets_to_update = [snippet for snippet in tagged_snippets_with_db_actions.snippets_to_update if snippet.id in prev_ids]
    snippets_to_delete = [snippet for snippet in tagged_snippets_with_db_actions.snippets_to_delete if snippet.id in prev_ids]
    return snippets_to_add, snippets_to_update, snippets_to_delete


def _execute_db_actions(collection, snippets_to_add:list[Snippet], snippets_to_update:list[Snippet], snippets_to_delete:list[Snippet]):
    if snippets_to_add:
        insert_to_index(
            collection=collection,
            documents=[snippet.text for snippet in snippets_to_add],
            metadatas=[{"date_of_event": snippet.date_of_event} if snippet.date_of_event else {"date_of_event": ""} for snippet in snippets_to_add]
        )
    if snippets_to_update:
        update_index(
            ids=[snippet.id for snippet in snippets_to_update],
            documents=[snippet.text for snippet in snippets_to_update],
        )
    if snippets_to_delete:
        delete_from_index(collection=collection, ids=[snippet.id for snippet in snippets_to_delete])


def maintain_index(collection, conversation_snippets: ConversationSnippets):
    all_related_snippets = []
    seen = set()
    for snippet in conversation_snippets.snippets:
        query_result = _retrieve_related_snippets(snippet)
        related_snippets = _process_related_snippets(query_result)
        for snippet in related_snippets:
            if snippet.id not in seen:
                all_related_snippets.append(snippet)
                seen.add(snippet.id)
    logger.info(f"Successfully retrieved related snippets. {all_related_snippets}")
    prev_ids = set([snippet.id for snippet in all_related_snippets])
    tag_snippets_with_db_actions = _tag_db_action_to_snippet(
        conversation_snippets=conversation_snippets,
        prior_related_snippets=all_related_snippets,
        model="o1-preview",
    )
    logger.info(f"Successfully tagged snippets with db actions. {tag_snippets_with_db_actions}")
    snippets_to_add, snippets_to_update, snippets_to_delete = _clean_tagged_snippets_with_db_actions(prev_ids,
                                                                                                     tag_snippets_with_db_actions)
    logger.info(
        f"Successfully cleaned tagged snippets with db actions, {snippets_to_add}, {snippets_to_update}, {snippets_to_delete}")
    # return snippets_to_add, snippets_to_update, snippets_to_delete
    _execute_db_actions(collection, snippets_to_add, snippets_to_update, snippets_to_delete)
    logger.info(f"Successfully executed db actions")


# maintain_index(
#     collection=collection,
#     conversation_snippets=ConversationSnippets(
#         snippets=[
#             Snippet(text="User only wants to be an astronaut now.", date_of_event="2024-11-20")
#         ]
#     )
# )

async def store(user_messages:list[str], assistant_messages:list[str]):
    conversation_snippets = extract_snippets_from_conversation(
        user_messages=user_messages,
        assistant_messages=assistant_messages
    )
    # determine_snippets_to_add_or_delete()
    # delete_documents_from_index()\
    try:
        maintain_index(collection=collection, conversation_snippets=conversation_snippets)
    except:
        insert_snippets_to_index(collection=collection, conversation_snippets=conversation_snippets)
    logger.info(f"There are now {collection.count()} documents in the index")

    update_knowledge(user_messages, assistant_messages, current_knowledge)

def recap():
    return current_knowledge.knowledge


insert_to_index(
    collection=collection,
    documents=[
        "user got laid off in 2024-11-10",
        "user is currently taking introduction to ML",
        "user is the sole breadwinner for a family of 4",
        "user has two young kids",
        "user is middle-income",
        "user is willing to upskill themselves or pivot industries/roles",
        "user was previously a business analyst",
    ]
)