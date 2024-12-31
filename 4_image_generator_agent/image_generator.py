from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent
from tools import web_search, video_creation
load_dotenv()

llm = ChatGroq(model="llama3-8b-8192")
researcher = Agent(
    role='Researcher',
    goal='Find relevant information',
    backstory='You are an AI research assistant',
    tools=[web_search]
)

video_creator = Agent(
    role='Video Creator',
    goal='Create engaging videos',
    backstory='You are an AI video production assistant',
    tools=[video_creation]
)

from crewai import Crew

crew = Crew(
    agents=[researcher, video_creator],
    tasks=[
        Task(
            description="Research a topic and create a video about it",
            agent=researcher
        ),
        Task(
            description="Create a video based on the research",
            agent=video_creator
        )
    ]
)

result = crew.kickoff()


