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
# 1. LLM
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
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
# 3. RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return the output.
    """

    if not isinstance(code, str):
        code = str(code)

    # Remove markdown code blocks
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
# 4. TEST CASE GENERATOR
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3 to 5 QA test scenarios.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 specific test scenarios for this coding task:

{task_description}

Include:
1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid cases when appropriate

For every test case include:
- Input
- Expected output

Return a numbered list.
Keep it concise.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return str(response.content)

    return str(response)


# ============================================================
# 5. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["task"]

    print("\n[Developer] Generating code...")

    prompt = f"""
You are a Python developer.

Solve this coding task:

{task}

Rules:

- Return ONLY Python code.
- Do not use markdown.
- Do not use ```python.
- Do NOT use input().
- Do NOT ask the user for input.
- Put the required sample data directly inside the program.
- Print the final result.
- Keep the program simple and clean.
"""

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

    # Remove markdown fences
    code = re.sub(
        r"```python\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = code.replace("```", "").strip()

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

    # --------------------------------------------------------
    # IMPORTANT:
    # Use normal string concatenation instead of a multiline
    # f-string to avoid syntax errors.
    # --------------------------------------------------------

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
        f"\n[Manager] Decision: {decision}"
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


# Compile

graph_app = workflow.compile()


print(
    "LangGraph workflow compiled successfully."
)


# ============================================================
# 10. API INPUT
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
            "Manager decision: "
            "store or another."
        )
    )


class CareerWorkflowOutput(BaseModel):

    task: str
    decision: str
    generated_code: str
    test_cases: str
    execution_output: str
    report: str
    next_step: str


# ============================================================
# 11. RUN LANGGRAPH
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
# 12. LANGSERVE CHAIN
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
# 13. FASTAPI
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


@app.get("/")
def home():

    return {
        "message": "LangGraph AI Coding Agent is running",
        "playground": "/career-agent/playground/"
    }


# ============================================================
# 14. SERVER
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
