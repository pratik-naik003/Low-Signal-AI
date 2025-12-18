from .test_generation_templates import Question
from pydantic import BaseModel,Field
from typing import List,Literal,Optional



class TopicList(BaseModel):
    topics:List[str]=Field(description="List of topics to be covered in the learning path")

class LearningPathInput(BaseModel):
    subject:str=Field(description="Subject for which the learning path is to be created")
    year_old:int=Field(gt=0,le=100,description="Age of the learner in years")
    preferred_language:Literal["en","hi","mr"]=Field(description="Preferred language for learning materials")
    focus_areas: List[str] = []

class Topic(BaseModel):
    topic_name:str=Field(description="Name of the topic")
    explaination:str=Field(description="Detailed explanation of the topic")
    practice_questions:List[Question]=Field(default=None,description="List of 2-3 easy practice questions for the topic for learning purposes")
    
class LearningPathOutPut(BaseModel):
    topics:list[Topic]= Field(description="List of topics to include in the learning path")
    additional_resources:Optional[List[str]]=Field(default=None,description="Additional resources for learning")
    
class TopicDetail(BaseModel):
    payload:LearningPathInput=Field(description="Input details for generating topic detail")
    topic_name:str=Field(description="Name of the topic")
    