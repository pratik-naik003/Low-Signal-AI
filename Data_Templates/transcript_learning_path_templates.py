from typing import Optional, List
from pydantic import BaseModel


class TranscriptLearningPathInput(BaseModel):
    transcript: str
    preferred_language: str = "en"


class Topic(BaseModel):
    topic_name: str
    explanation: str
    practice_questions: List[str]


class TopicList(BaseModel):
    topics: List[str]


class LearningPathOutPut(BaseModel):
    topics: List[Topic]
    additional_resources: Optional[List[str]] = None


class TopicDetail(BaseModel):
    payload: TranscriptLearningPathInput
    topic_name: str
