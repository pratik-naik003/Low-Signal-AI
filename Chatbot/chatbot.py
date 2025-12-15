from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",streaming=True)


def Ai_stream(question:str):
    for chunk in model.stream(question):
        if(chunk.content):
            yield chunk.content