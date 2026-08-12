%%writefile app.py

import os
import re
import sys
import subprocess
from typing import TypedDict, Optional, List

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

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

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
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Safely execute generated Python code in a separate process.
    Used for testing generated code.
    """

    if not isinstance(code, str):
        code = str(code)

    # Remove markdown code fences if Gemini accidentally returns them
    clean_code = re.sub(r"```python\s*", "", code, flags=re.IGNORECASE)
    clean_code = clean_code.replace("```", "").strip()

    try:
        result = subprocess.run(
            [sys.executable, "-c", clean_code],
            input="2\n",
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

        return (
            f"Execution Error:\n{error}"
            if error
            else "Program exited with an error."
        )

    except subprocess.TimeoutExpired:
        return "Execution Error: Program exceeded the 5 second timeout."

    except Exception as e:
        return f"Execution Error: {str(e)}"


@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3-5 test scenarios for the coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 specific test scenarios for this coding task:

{task_description}

Include:
1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid input cases when appropriate

For every test case provide:
- Input
- Expected output

Keep the response concise and use a numbered list.
"""

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ============================================================
# 4. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["task"]

    print("\n[Developer] Generating Python solution...")

    prompt = f"""
You are a Python software developer.

Solve this coding task:

{task}

Important requirements:

- Return ONLY Python code.
- Do not use markdown.
- Do not use ``` fences.
- Do NOT use input().
- Do NOT require interactive terminal input.
- Put a sample value directly into a variable so the program can run automatically.
- The program must print its result.
- Keep the code simple and clean.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        code_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    code_parts.append(item.get("text", ""))
                elif "text" in item:
                    code_parts.append(item["text"])
            else:
                code_parts.append(str(item))

        code = "\n".join(code_parts)

    else:
        code = str(content)

    # Remove markdown fences if present
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
        "messages": [HumanMessage(content=task)]
    }


# ============================================================
# 5. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    print("\n[Tester] Generating tests and executing code...")

    task = state["task"]
    code = state["code"]

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):
        cases = "\n".join(str(x) for x in test_cases)
    else:
        cases = str(test_cases)

    # Execute generated code
    execution_output = run_python_code.invoke({
        "code": code
    })

    report = f"""
### CODING TASK

{task}

### GENERATED CODE

```python
{code}
