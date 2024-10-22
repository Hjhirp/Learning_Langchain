from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ["You are a twitter influencer grading a tweet. Generate critique and recommendations for user.",
            "Always provide detailed recommendations, including requests for length, virality etc."],
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prmompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ["You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate a tweet that will go viral.",
            "If user provides critique, respond with a revised tweet."],
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generation_chain = generation_prmompt | llm
reflect_chain = reflection_prompt | llm

