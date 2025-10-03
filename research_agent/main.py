# from dotenv import load_dotenv
# from pydantic import BaseModel
# # from langchain_openai import ChatOpenAI
# # from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent,AgentExecutor
# from tools import search_tool,wiki_tool,save_tool

# load_dotenv()

# class ResearchResponse(BaseModel):
#     topic : str
#     summary : str
#     sources : list[str]
#     tools_used : list[str]
    



# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# # llm = ChatAnthropic(model = "Claude Sonnet 4.x")
# # response = llm.invoke("tell me about AI research")
# # print(response)

# parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a research assistant that will help generate a research paper.
#             Answer the user query and use neccessary tools. 
#             Wrap the output in this format and provide no other text\n{format_instructions}
#             """,
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{query}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())

# tools = [search_tool,wiki_tool,save_tool]
# agent = create_tool_calling_agent(
#     llm = llm,
#     prompt = prompt,
#     tools = tools
# )

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# query = input("What can i help you to research today?")

# raw_response = agent_executor.invoke({"query": query})


# # print(raw_response)

# try:
#     structured_response = parser.parse(raw_response.get("output"))
#     print(structured_response)
# except Exception as e:
#     print("Error parsing response:", e,"Raw response:", raw_response)



import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.exceptions import OutputParserException
from tools import search_tool, wiki_tool, save_tool 

# --- 1. Environment and Constants ---
load_dotenv()

# --- 2. Pydantic Model ---
class ResearchResponse(BaseModel):
    """Schema for the final research output."""
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
LLM_MODEL = "gemini-2.5-flash" 

TOOLS = [search_tool,wiki_tool,save_tool] 

# --- 4. Initialization ---
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1) 

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- 5. Prompt Refinement ---
# Added stronger directives for tool use, response format, and tone.
prompt_template = """
    You are a highly efficient and accurate **Research Assistant**. 
    Your primary goal is to answer the user's query comprehensively.

    **INSTRUCTIONS**:
    1. **Tool Use**: Use the provided tools (e.g., search_tool, wiki_tool) to gather necessary, current, and factual information. Only use tools when needed.
    2. **Formatting**: After all necessary steps, you **MUST** wrap your final answer in a proper text format and save to file.
    3.**Deep search**: Do deep search on every query, do not stop at the first result.
    4.**Tone**: Maintain a professional and neutral tone in your responses.
    3. **Completeness**: Fill all fields in the JSON structure. If no tools were used, the 'tools_used' list must be empty.

    {format_instructions}
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# --- 6. Agent Setup ---
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=TOOLS
)

# Set handle_parsing_errors=True for robust operation
agent_executor = AgentExecutor(
    agent=agent, 
    tools=TOOLS, 
    verbose=True,
    handle_parsing_errors=True # Better error handling during tool calls
)

# --- 7. Main Execution ---
if __name__ == "__main__":
    query = input("What can I help you to research today? ")
    
    print("\n--- Running Research Agent ---")
    try:
        # Invoke the agent
        raw_response = agent_executor.invoke({"query": query, "chat_history": []})
        raw_output_text = raw_response.get("output")
        
        # Parse the output text using the Pydantic parser
        structured_response = parser.parse(raw_output_text)
        
        print("\n--- ✅ Structured Research Output ---")
        print(structured_response.model_dump_json(indent=2))
        
    except OutputParserException as e:
        print(f"\n--- ❌ Output Parsing Error ---")
        print(f"Failed to parse the final response into the Pydantic model. Check the LLM's raw output.")
        print(f"Error: {e}")
        print(f"Raw response: {raw_response}")
        
    except Exception as e:
        print(f"\n--- ❌ General Error ---")
        print(f"An unexpected error occurred: {e}")