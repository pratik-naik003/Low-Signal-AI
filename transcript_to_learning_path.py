import json
from dotenv import load_dotenv

from langchain_cerebras import ChatCerebras
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    JsonOutputParser,
)


from Data_Templates.transcript_learning_path_templates import (
    TranscriptLearningPathInput,
    Topic,
    TopicList,
    LearningPathOutPut,
    TopicDetail,
)

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507")
model_stream = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507",streaming=True)


topic_list_parser = PydanticOutputParser(pydantic_object=TopicList)
topic_parser = PydanticOutputParser(pydantic_object=Topic)


topic_planner_prompt = PromptTemplate(
    template="""
You are an expert curriculum designer.

Task:
Analyze the given YouTube transcript and generate a structured learning path.

Transcript:
{transcript}

Preferred language: {preferred_language}

Strict rules:
- Generate exactly 6 to 8 topics
- Topics must be ordered from basic to advanced
- Topic names must be short, clear, and non-overlapping
- Topics must be derived ONLY from the transcript
- Do NOT include explanations, numbering, or examples
- Use ONLY the preferred language:
  - en → English
  - hi → Hindi
  - mr → Marathi

Output rules:
- Return ONLY valid JSON
- Must strictly follow the schema
- Do NOT add extra fields or text

{format_instructions}
""",
    input_variables=["transcript", "preferred_language"],
    partial_variables={
        "format_instructions": topic_list_parser.get_format_instructions()
    },
)

topic_expander_prompt = PromptTemplate(
    template="""
You are an expert tutor.

Task:
Explain the given topic using ONLY the transcript content
and generate beginner-friendly practice questions.

Transcript:
{transcript}

Topic:
{topic_name}

Preferred language: {preferred_language}

Strict rules:
- Explanation must be clear, structured, and detailed
- Create exactly 2 or 3 easy practice questions
- Do NOT reference other topics
- Do NOT add answers
- Use ONLY transcript information
- Use ONLY the preferred language

Output rules:
- Return ONLY valid JSON
- Must strictly follow the schema

{format_instructions}
""",
    input_variables=["transcript", "preferred_language", "topic_name"],
    partial_variables={
        "format_instructions": topic_parser.get_format_instructions()
    },
)


topic_planner_chain = topic_planner_prompt | model | topic_list_parser
topic_expander_chain = topic_expander_prompt | model | topic_parser

topic_expander_stream_chain = (
    topic_expander_prompt | model_stream | JsonOutputParser()
)


def create_learning_path_from_transcript(
    payload: TranscriptLearningPathInput
) -> LearningPathOutPut:
    
    topic_list = topic_planner_chain.invoke(payload.model_dump())

    topics_detailed = topic_expander_chain.batch([
        {
            "transcript": payload.transcript,
            "preferred_language": payload.preferred_language,
            "topic_name": topic,
        }
        for topic in topic_list.topics
    ])

    return LearningPathOutPut(
        topics=topics_detailed,
        additional_resources=None
    )


def create_topic_list_from_transcript(
    payload: TranscriptLearningPathInput
) -> TopicList:
    return topic_planner_chain.invoke(payload.model_dump())


def create_topic_detail_from_transcript(
    payload: TopicDetail
) -> Topic:
    return topic_expander_chain.invoke(
        {
            "transcript": payload.payload.transcript,
            "preferred_language": payload.payload.preferred_language,
            "topic_name": payload.topic_name,
        }
    )


def topic_detail_event_stream(payload: TopicDetail):
    """
    Streams explanation incrementally and questions at the end
    (SSE compatible)
    """

    input_data = {
        "transcript": payload.payload.transcript,
        "preferred_language": payload.payload.preferred_language,
        "topic_name": payload.topic_name,
    }

    last_sent_length = 0
    final_data = {}

    try:
        for chunk in topic_expander_stream_chain.stream(input_data):
            final_data = chunk

            if "explanation" in chunk and chunk["explanation"]:
                current_text = chunk["explanation"]

                if len(current_text) > last_sent_length:
                    new_content = current_text[last_sent_length:]
                    last_sent_length = len(current_text)

                    yield f"data: {json.dumps({
                        'type': 'explanation_chunk',
                        'data': new_content
                    })}\n\n"

        if "practice_questions" in final_data:
            for q in final_data["practice_questions"]:
                yield f"data: {json.dumps({
                    'type': 'question',
                    'data': q
                })}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({
            'type': 'error',
            'message': str(e)
        })}\n\n"
