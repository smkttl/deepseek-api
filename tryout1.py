from openai import OpenAI
import json
import os
import zipfile

# 1. Initialize the client
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-dummy-key"
)

# ---------------------------------------------------------------- tools
def create_folder(path: str) -> dict:
    os.makedirs(path, exist_ok=True)
    return {"status": "ok", "path": path, "message": f"Folder ready: {path}"}

def write_file(path: str, content: str) -> dict:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "path": path, "bytes_written": len(content.encode("utf-8"))}

def zip_folder(src_dir: str, zip_path: str) -> dict:
    """Zip everything under src_dir into zip_path (recursively)."""
    parent = os.path.dirname(zip_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arc  = os.path.relpath(full, src_dir)
                zf.write(full, arc)
                count += 1
    return {"status": "ok", "zip": zip_path, "files_included": count}

TOOL_EXECUTORS = {
    "create_folder": create_folder,
    "write_file": write_file,
    "zip_folder": zip_folder,
}

# ---------------------------------------------------------------- spec
TOOLS_SPEC = """
Available tools:

1. create_folder
   args: { "path": "<string>" }
   Creates a folder (and any parent folders).

2. write_file
   args: { "path": "<string>", "content": "<string>" }
   Writes UTF-8 text to a file (creates parent folder if needed).
   Use this for the documentation, the .c program, and the input/output
   files.

3. zip_folder
   args: { "src_dir": "<string>", "zip_path": "<string>" }
   Recursively zips everything under src_dir into a single .zip file.
"""

# ---------------------------------------------------------------- prompt
# Choose a student name / roll number here. The submission ZIP name
# must follow Name_btech_roll_Assignment1 (e.g. Ajay_Btech_9_Assignment1).
STUDENT_NAME = "Adnan_Aftab"
STUDENT_ROLL = "Btech_23"
ZIP_NAME     = f"{STUDENT_NAME}_{STUDENT_ROLL}_Assignment1.zip"
ROOT_DIR     = f"{STUDENT_NAME}_{STUDENT_ROLL}_Assignment1"

PROMPT = f"""You are a tool-using agent. You MUST respond with ONLY a JSON array
(no prose, no markdown fences) where each element is an object of the form:

  {{ "tool": "<tool_name>", "args": {{ ... }} }}

The array is executed top-to-bottom. Produce the tool calls needed to complete
the user's request below. Do NOT put the final answer in prose — put everything
into tool calls.

{TOOLS_SPEC}

USER REQUEST:
Prepare a complete Assignment 1 submission for the topic:

    "Lexical Analysis of a C program"

The submission must be a single ZIP file named "{ZIP_NAME}" and the ZIP must
contain a root folder "{ROOT_DIR}" with exactly these parts:

  A) documentation.txt  -- Must contain ALL of the following sections:
       (1) Problem statement
       (2) Program
       (3) Input (format + one example)
       (4) Output (format + example output, including each intermediate
           step such as the list of tokens, lexemes, token types, line
           numbers, etc.)
     Make the documentation complete and self-contained.

  B) lexer.c  -- A working C source file that performs lexical analysis of a
     C program: reads a .c file, and prints every token (keyword, identifier,
     constant, operator, punctuation, etc.) one per line, plus a summary.
     Keep it plain C (stdio.h, ctype.h, string.h).

  C) input.c  -- A small sample C program used as input to lexer.c.
     It should exercise keywords, identifiers, numbers, strings, comments,
     operators, and punctuation.

  D) output.txt -- The exact expected stdout produced by running lexer.c on
     input.c. Include intermediate information: every token with its
     category and line number, then a token-count summary.

Files A–D must live inside the folder "{ROOT_DIR}/" and must be written with
the write_file tool. Then use zip_folder to zip that folder into
"{ZIP_NAME}" in the current directory.

Order the tool calls:
  create_folder("{ROOT_DIR}")
  write_file("{ROOT_DIR}/documentation.txt", ...)
  write_file("{ROOT_DIR}/lexer.c", ...)
  write_file("{ROOT_DIR}/input.c", ...)
  write_file("{ROOT_DIR}/output.txt", ...)
  zip_folder("{ROOT_DIR}", "{ZIP_NAME}")
"""

# ---------------------------------------------------------------- runner
def run_tool_calls(calls):
    results = []
    for i, call in enumerate(calls):
        name = call.get("tool")
        args = call.get("args", {}) or {}
        preview = json.dumps(args)
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"[{i+1}] {name}({preview})")
        fn = TOOL_EXECUTORS.get(name)
        if fn is None:
            results.append({"index": i, "tool": name,
                            "result": {"status": "error", "message": "unknown tool"}})
            continue
        try:
            res = fn(**args)
        except Exception as e:
            res = {"status": "error", "message": str(e)}
        results.append({"index": i, "tool": name, "result": res})
        print(f"    -> {res}")
    return results

def test_chat():
    print("Sending single-shot tool-planning request...")
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": PROMPT}],   # single message
        )

        raw = response.choices[0].message.content
        print("\n--- Raw Response (truncated) ---")
        print(raw[:800] + ("..." if len(raw) > 800 else ""))

        # Strip possible markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

        calls = json.loads(clean)
        if isinstance(calls, dict) and "calls" in calls:
            calls = calls["calls"]
        if not isinstance(calls, list):
            raise ValueError("Model did not return a JSON array of tool calls.")

        print(f"\n--- Executing {len(calls)} tool call(s) ---")
        results = run_tool_calls(calls)

        print("\n--- Summary ---")
        print(json.dumps(results, indent=2))
        print(f"\nSubmission ready: ./{ZIP_NAME}")

    except json.JSONDecodeError as e:
        print(f"Failed to parse model output as JSON: {e}")
    except Exception as e:
        print(f"\nError calling API: {e}")
        print("Is your server running on port 8000?")

if __name__ == "__main__":
    test_chat()