import os
import sys
import io
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLEAPI") or os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLEAPI or GOOGLE_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

llm = llm_flash


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict, total=False):

    messages: List[BaseMessage]

    next_step: Optional[str]

    code: Optional[str]

    report: Optional[str]

    task: Optional[str]

    decision: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return the standard output
    or error trace.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate specific test scenarios for a given coding task.
    """

    prompt = (
        f"You are a Senior QA Engineer.\n\n"
        f"Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task:\n\n"
        f"{task_description}\n\n"
        f"Include standard cases and edge cases.\n"
        f"Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ============================================================
# 4. GRAPH NODES
# ============================================================

def task_input_node(state: CrewState):

    print("\n" + "=" * 50)
    print("--- NEW TASK INITIALIZATION ---")

    # In the deployed API, the task comes from the request.
    user_task = state.get("task", "")

    if not user_task:
        return {
            "next_step": "exit"
        }

    print(f"Task received: {user_task}")

    return {
        "messages": [
            HumanMessage(content=user_task)
        ],
        "next_step": "developer"
    }


# ------------------------------------------------------------
# Developer
# ------------------------------------------------------------

def real_time_developer(state: CrewState):

    print("\n[Developer] Writing dynamic code using LLM...")

    task = state["messages"][-1].content

    dev_prompt = (
        f"Write a clean Python script to solve this coding task:\n\n"
        f"{task}\n\n"
        f"Only return the Python code. "
        f"Do not include explanations or markdown."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    code_parts.append(
                        item.get("text", "")
                    )

                elif "text" in item:
                    code_parts.append(
                        item.get("text", "")
                    )

            else:
                code_parts.append(str(item))

        code_str = "".join(code_parts)

    else:

        code_str = str(content)

    # Remove accidental markdown fences
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print(code_str)

    return {
        "code": code_str
    }


# ------------------------------------------------------------
# Tester
# ------------------------------------------------------------

def real_time_tester(state: CrewState):

    print(
        "\n[Tester] Generating dynamic tests "
        "and executing code..."
    )

    task = state["messages"][-1].content

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):

        cases_parts = []

        for item in test_cases:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    cases_parts.append(
                        item.get("text", "")
                    )

                elif "text" in item:
                    cases_parts.append(
                        item.get("text", "")
                    )

            else:
                cases_parts.append(str(item))

        cases_str = "".join(cases_parts)

    else:

        cases_str = str(test_cases)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Compile report
    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


# ------------------------------------------------------------
# Manager
# ------------------------------------------------------------

def manager_decision_node(state: CrewState):

    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : TEST REPORT ---")

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print("=" * 50)

    # In the deployed API, there is no input().
    # The decision comes from the API request.

    decision = state.get(
        "decision",
        "store"
    )

    decision = decision.lower().strip()

    if decision == "store":

        return {
            "next_step": "archiver"
        }

    # For a web API, "another" ends this run.
    # The user can submit another request from the Playground.

    return {
        "next_step": "exit"
    }


# ------------------------------------------------------------
# Archiver
# ------------------------------------------------------------

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] Task stored successfully. "
        "Closing workflow."
    )

    return {
        "next_step": "exit"
    }


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

rt_workflow = StateGraph(CrewState)


rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START → task_input

rt_workflow.add_edge(
    START,
    "task_input"
)


# ============================================================
# 6. INPUT ROUTING
# ============================================================

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":

        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ============================================================
# 7. SEQUENTIAL FLOW
# ============================================================

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ============================================================
# 8. MANAGER ROUTING
# ============================================================

def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":

        return "archiver"

    return END


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ============================================================
# 9. ARCHIVER → END
# ============================================================

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile graph

rt_app = rt_workflow.compile()


print(
    "LangGraph workflow compiled successfully."
)


# ============================================================
# 10. API INPUT SCHEMA
# ============================================================

class GraphInput(BaseModel):

    task: str = Field(
        ...,
        description=(
            "Coding task to give to the developer agent. "
            "Example: Write a Python program to check whether "
            "a number is prime."
        )
    )

    decision: str = Field(
        default="store",
        description=(
            "Manager decision. Use 'store' to complete "
            "the workflow or 'another' to end this run."
        )
    )


# ============================================================
# 11. API RUNNER
# ============================================================

def run_langgraph(payload) -> dict:

    # LangServe sends a dictionary.
    # Therefore use payload["task"].

    task = payload["task"]

    decision = payload.get(
        "decision",
        "store"
    )

    initial_state: CrewState = {

        "messages": [],

        "next_step": "developer",

        "code": None,

        "report": None,

        "task": task,

        "decision": decision
    }

    try:

        result = rt_app.invoke(
            initial_state,
            config={
                "recursion_limit": 50
            }
        )

        return {
            "task": task,
            "decision": decision,
            "generated_code": result.get(
                "code",
                ""
            ),
            "report": result.get(
                "report",
                ""
            ),
            "next_step": result.get(
                "next_step",
                "exit"
            )
        }

    except Exception as e:

        return {
            "task": task,
            "decision": decision,
            "generated_code": result.get(
                "code",
                ""
            ) if "result" in locals() else "",
            "report": (
                "Workflow execution failed:\n"
                f"{str(e)}"
            ),
            "next_step": "exit"
        }


# ============================================================
# 12. LANGSERVE RUNNABLE
# ============================================================

career_chain = (
    RunnableLambda(run_langgraph)
    .with_types(
        input_type=GraphInput,
        output_type=dict
    )
)


# ============================================================
# 13. FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph Real-Time Developer Tester",
    description=(
        "LangGraph workflow with Developer, Tester, "
        "Manager and Archiver nodes."
    )
)


add_routes(
    app,
    career_chain,
    path="/langgraph-agent",
    playground_type="default"
)


# ============================================================
# 14. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
