import json
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

from Data_Templates.transcript_learning_path_templates import (
    TranscriptLearningPathInput,
    Topic,
    TopicList,
    LearningPathOutPut,
    TopicDetail,
)

load_dotenv()

model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507")
model_stream = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",
    streaming=True
)

topic_list_parser = PydanticOutputParser(pydantic_object=TopicList)
topic_parser = PydanticOutputParser(pydantic_object=Topic)

topic_planner_prompt = PromptTemplate(
    template="""
Generate a learning path from this transcript.

Transcript:
{transcript}

Language: {preferred_language}

Rules:
- 6–8 topics
- basic → advanced
- short names
- ONLY transcript info
- JSON only

{format_instructions}
""",
    input_variables=["transcript", "preferred_language"],
    partial_variables={"format_instructions": topic_list_parser.get_format_instructions()}
)

topic_expander_prompt = PromptTemplate(
    template="""
Explain this topic using the transcript.

Transcript:
{transcript}

Topic:
{topic_name}

Language: {preferred_language}

Rules:
- clear explanation
- 2–3 practice questions
- JSON only

{format_instructions}
""",
    input_variables=["transcript", "preferred_language", "topic_name"],
    partial_variables={"format_instructions": topic_parser.get_format_instructions()}
)

topic_planner_chain = topic_planner_prompt | model | topic_list_parser
topic_expander_chain = topic_expander_prompt | model | topic_parser
topic_stream_chain = topic_expander_prompt | model_stream | JsonOutputParser()


def create_learning_path_from_transcript(payload: TranscriptLearningPathInput):
    topic_list = topic_planner_chain.invoke(payload.model_dump())

    topics = topic_expander_chain.batch([
        {
            "transcript": payload.transcript,
            "preferred_language": payload.preferred_language,
            "topic_name": t
        }
        for t in topic_list.topics
    ])

    return LearningPathOutPut(topics=topics)


def topic_detail_event_stream(payload: TopicDetail):
    input_data = {
        "transcript": payload.payload.transcript,
        "preferred_language": payload.payload.preferred_language,
        "topic_name": payload.topic_name,
    }

    last_len = 0
    final = {}

    for chunk in topic_stream_chain.stream(input_data):
        final = chunk
        if "explanation" in chunk:
            new = chunk["explanation"][last_len:]
            last_len = len(chunk["explanation"])
            if new:
                yield f"data: {json.dumps({'type':'text','data':new})}\n\n"

    for q in final.get("practice_questions", []):
        yield f"data: {json.dumps({'type':'question','data':q})}\n\n"

    yield "data: {\"type\":\"done\"}\n\n"
