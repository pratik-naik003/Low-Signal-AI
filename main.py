from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# ---------------- CHAT ----------------
from Chatbot.chatbot import Ai_stream

# ---------------- TEST GENERATION ----------------
from testGenerator.generate_test import generate_test_ai
from Data_Templates.test_generation_templates import (
    TestGenInput,
    TestGenOutput
)

# ---------------- NORMAL LEARNING PATH ----------------
from Data_Templates.learning_path_templates import (
    LearningPathInput,
    LearningPathOutPut,
    TopicList,
    Topic,
    TopicDetail
)

from learningpath import (
    create_learning_path,
    create_topic_list,
    create_topic_detail
)

# ---------------- TRANSCRIPT LEARNING PATH ----------------
from Data_Templates.transcript_learning_path_templates import (
    TranscriptLearningPathInput,
    TopicDetail as TranscriptTopicDetail,
    TopicList as TranscriptTopicList,
    LearningPathOutPut as TranscriptLearningPathOutPut,
    Topic as TranscriptTopic
)

from transcript_to_learning_path import (
    create_learning_path_from_transcript,
    create_topic_list_from_transcript,
    create_topic_detail_from_transcript,
    topic_detail_event_stream as transcript_topic_detail_event_stream
)

# ---------------- YOUTUBE TRANSCRIPT ----------------
from youtube.transcript import get_youtube_transcript


# ---------------- FASTAPI APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- BASIC SCHEMAS ----------------
class Query(BaseModel):
    question: str


class YouTubeInput(BaseModel):
    url: str
    preferred_language: str = "en"


# ---------------- HEALTH ----------------
@app.get("/")
def health():
    return {"Status": "Ok"}


# ---------------- CHAT STREAM ----------------
@app.get("/chat/stream")
def chat_stream(question: str):
    def event_generator():
        for token in Ai_stream(question):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# ---------------- TEST GENERATION ----------------
@app.post("/test/generate", response_model=TestGenOutput)
def generate_test(payload: TestGenInput):
    return generate_test_ai(payload)


# ================= NORMAL LEARNING PATH =================

@app.post("/learning_path/generate", response_model=LearningPathOutPut)
def generate_learning_path(payload: LearningPathInput):
    return create_learning_path(payload)


@app.post("/learning_path/generate/topic_list", response_model=TopicList)
def generate_topic_list(payload: LearningPathInput):
    return create_topic_list(payload)


@app.post("/learning_path/generate/topic_detail", response_model=Topic)
def generate_topic_detail(payload: TopicDetail):
    return create_topic_detail(payload)


# ================= TRANSCRIPT LEARNING PATH (RAW TRANSCRIPT) =================

@app.post(
    "/transcript/learning_path/generate",
    response_model=TranscriptLearningPathOutPut
)
def generate_learning_path_from_transcript_api(
    payload: TranscriptLearningPathInput
):
    return create_learning_path_from_transcript(payload)


@app.post(
    "/transcript/learning_path/generate/topic_list",
    response_model=TranscriptTopicList
)
def generate_transcript_topic_list(
    payload: TranscriptLearningPathInput
):
    return create_topic_list_from_transcript(payload)


@app.post(
    "/transcript/learning_path/generate/topic_detail",
    response_model=TranscriptTopic
)
def generate_transcript_topic_detail(
    payload: TranscriptTopicDetail
):
    return create_topic_detail_from_transcript(payload)


@app.post("/transcript/learning_path/generate/topic_detail/stream")
def stream_transcript_topic_detail(
    payload: TranscriptTopicDetail
):
    return StreamingResponse(
        transcript_topic_detail_event_stream(payload),
        media_type="text/event-stream"
    )


# ================= YOUTUBE URL → LEARNING PATH =================

@app.post(
    "/youtube/learning_path/generate",
    response_model=TranscriptLearningPathOutPut
)
def generate_learning_path_from_youtube(
    payload: YouTubeInput
):
    transcript = get_youtube_transcript(payload.url)

    transcript_payload = TranscriptLearningPathInput(
        transcript=transcript,
        preferred_language=payload.preferred_language
    )

    return create_learning_path_from_transcript(transcript_payload)
