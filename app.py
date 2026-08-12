import os
import re
import sys
import subprocess
from typing import TypedDict, Optional, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

GOOGLEAPI = os.environ.get("GOOGLEAPI")

if not GOOGLEAPI:
    raise ValueError(
        "GOOGLEAPI environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLEAPI,
    temperature=0
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict, total=False):
    task: str
    decision: str
    messages: List[BaseMessage]
    code: str
    test_cases: str
    execution_output: str
    report: str
    next_step: Optional[str]


# ============================================================
# 3. TOOL: RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return the output.
    """

    if not isinstance(code, str):
        code = str(code)

    # Remove Markdown code fences if Gemini returns them
    clean_code = re.sub(
        r"```python\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    clean_code = clean_code.replace("```", "").strip()

    try:

        result = subprocess.run(
            [sys.executable, "-c", clean_code],
            input="",
            text=True,
            capture_output=True,
            timeout=5
        )

        if result.returncode == 0:

            output = result.stdout.strip()

            if output:
                return output

            return "Program executed successfully with no output."

        error = result.stderr.strip()

        if error:
            return f"Execution Error:\n{error}"

        return "Program exited with an error."

    except subprocess.TimeoutExpired:

        return (
            "Execution Error: "
            "Program exceeded the 5 second timeout."
        )

    except Exception as e:

        return f"Execution Error: {str(e)}"


# ============================================================
# 4. TOOL: GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3 to 5 QA test scenarios for the coding task.
    """

    prompt = (
        "You are a Senior QA Engineer.\n\n"
        "Generate 3 to 5 specific test scenarios for this "
        "coding task:\n\n"
        + task_description
        + "\n\n"
        "Include:\n"
        "1. Normal cases\n"
        "2. Boundary cases\n"
        "3. Edge cases\n"
        "4. Invalid cases when appropriate\n\n"
        "For every test case include:\n"
        "- Input\n"
        "- Expected output\n\n"
        "Return a numbered list and keep it concise."
    )

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return str(response.content)

    return str(response)


# ============================================================
# 5. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["task"]

    print("\n[Developer] Generating Python code...")

    prompt = (
        "You are a Python developer.\n\n"
        "Solve this coding task:\n\n"
        + task
        + "\n\n"
        "Rules:\n"
        "- Return ONLY Python code.\n"
        "- Do not return explanations.\n"
        "- Do not use Markdown.\n"
        "- Do not use ```python.\n"
        "- Do NOT use input().\n"
        "- Do NOT ask the user for input.\n"
        "- Put all required sample data directly inside the program.\n"
        "- Print the final result.\n"
        "- Keep the program simple and clean."
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    parts.append(
                        item.get("text", "")
                    )

                elif "text" in item:
                    parts.append(
                        item.get("text", "")
                    )

            else:
                parts.append(str(item))

        code = "\n".join(parts)

    else:

        code = str(content)

    # Remove Markdown code fences
    code = re.sub(
        r"```python\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = code.replace("```", "").strip()

    print("\nGenerated Code:")
    print(code)

    return {
        "code": code,
        "messages": [
            HumanMessage(content=task)
        ]
    }


# ============================================================
# 6. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    print("\n[Tester] Generating test cases...")

    task = state["task"]
    code = state["code"]

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):

        cases = "\n".join(
            str(item)
            for item in test_cases
        )

    else:

        cases = str(test_cases)

    # Execute generated code
    print("[Tester] Executing generated code...")

    execution_output = run_python_code.invoke(
        {
            "code": code
        }
    )

    # Build report without triple-quoted f-strings
    report = (
        "### CODING TASK\n\n"
        + task
        + "\n\n"
        + "### GENERATED CODE\n\n"
        + code
        + "\n\n"
        + "### EXECUTION OUTPUT\n\n"
        + execution_output
        + "\n\n"
        + "### TEST SCENARIOS\n\n"
        + cases
    )

    print("\n[Tester] Testing completed.")

    return {
        "test_cases": cases,
        "execution_output": execution_output,
        "report": report
    }


# ============================================================
# 7. MANAGER NODE
# ============================================================

def manager_node(state: CrewState):

    decision = state.get(
        "decision",
        "store"
    )

    decision = decision.lower().strip()

    print(
        "\n[Manager] Decision: "
        + decision
    )

    if decision == "store":

        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "exit"
    }


# ============================================================
# 8. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] "
        "Task stored successfully."
    )

    return {
        "next_step": "exit"
    }


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(CrewState)


# Add nodes
workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)

workflow.add_node(
    "archiver",
    archiver_node
)


# START → Developer
workflow.add_edge(
    START,
    "developer"
)


# Developer → Tester
workflow.add_edge(
    "developer",
    "tester"
)


# Tester → Manager
workflow.add_edge(
    "tester",
    "manager"
)


# ============================================================
# MANAGER ROUTING
# ============================================================

def route_manager(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return END


workflow.add_conditional_edges(
    "manager",
    route_manager
)


# Archiver → END
workflow.add_edge(
    "archiver",
    END
)


# Compile graph
graph_app = workflow.compile()

print(
    "LangGraph workflow compiled successfully."
)


# ============================================================
# 10. API INPUT SCHEMA
# ============================================================

class CareerWorkflowInput(BaseModel):

    task: str = Field(
        ...,
        description=(
            "Coding task for the AI developer."
        )
    )

    decision: str = Field(
        default="store",
        description=(
            "Manager decision: store or another."
        )
    )


# ============================================================
# 11. API OUTPUT SCHEMA
# ============================================================

class CareerWorkflowOutput(BaseModel):

    task: str
    decision: str
    generated_code: str
    test_cases: str
    execution_output: str
    report: str
    next_step: str


# ============================================================
# 12. RUN LANGGRAPH WORKFLOW
# ============================================================

def run_career_workflow(
    payload: CareerWorkflowInput
) -> dict:

    initial_state: CrewState = {

        "task": payload.task,

        "decision": payload.decision,

        "messages": [
            HumanMessage(
                content=payload.task
            )
        ]
    }

    try:

        result = graph_app.invoke(
            initial_state
        )

        return {

            "task": result.get(
                "task",
                payload.task
            ),

            "decision": payload.decision,

            "generated_code": result.get(
                "code",
                ""
            ),

            "test_cases": result.get(
                "test_cases",
                ""
            ),

            "execution_output": result.get(
                "execution_output",
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

            "task": payload.task,

            "decision": payload.decision,

            "generated_code": "",

            "test_cases": "",

            "execution_output": "",

            "report": (
                "Workflow Error:\n"
                + str(e)
            ),

            "next_step": "exit"
        }


# ============================================================
# 13. LANGSERVE CHAIN
# ============================================================

career_chain = (
    RunnableLambda(
        run_career_workflow
    )
    .with_types(
        input_type=CareerWorkflowInput,
        output_type=CareerWorkflowOutput
    )
)


# ============================================================
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph AI Coding Agent"
)


add_routes(
    app,
    career_chain,
    path="/career-agent",
    playground_type="default"
)


# ============================================================
# 15. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph AI Coding Agent is running",
        "playground": "/career-agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 16. START SERVER
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
