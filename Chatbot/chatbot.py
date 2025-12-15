from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv

load_dotenv()

model = ChatCerebras(model="llama-3.3-70b",streaming=True)


def Ai_stream(question:str):
    for chunk in model.stream(question):
        if(chunk.content):
            yield chunk.content