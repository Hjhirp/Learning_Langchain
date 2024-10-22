import json
from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool])

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]

    # print("State", state[-1])

    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        dump = json.dumps(parsed_call, indent=4)
        print("Parsed Call: ", dump)

        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json",
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])
    # print("Tool Invocations: ", tool_invocations)
    outputs = tool_executor.batch(tool_invocations)
    # print("Outputs: ", outputs)

    outputs_map = defaultdict(dict)
    for id_, output, invocations in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocations.tool_input] = output

    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_))

    return tool_messages

if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[x
            "AI-Powered SOC solutions 2024",
            "Startups in autonomous SOC space 2024",
            "Investment in AI cybersecurity startups 2024",
        ],
        id="call_IMChTVxkPZV2gnMh299sVjf6",
    )

    raw_res = execute_tools(
        state=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_IMChTVxkPZV2gnMh299sVjf6",
                    }
                ],
            ),
        ]
    )
    # print(raw_res)