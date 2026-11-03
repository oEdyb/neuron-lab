import json
import subprocess
import tempfile


class ClaudeFailed(Exception):
    pass


def ask(prompt, schema, model):
    """Calls `claude -p` from an empty folder with no tools, so it sees only the prompt."""
    command = ["claude", "-p", prompt, "--model", model, "--tools", "",
               "--output-format", "json", "--json-schema", json.dumps(schema),
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
    with tempfile.TemporaryDirectory() as empty:
        done = subprocess.run(command, cwd=empty, capture_output=True, text=True, timeout=180,
                              stdin=subprocess.DEVNULL)
    if done.returncode:
        raise ClaudeFailed(f"claude -p exited {done.returncode}: {done.stderr.strip()[:300]}")
    return json.loads(done.stdout)["structured_output"]
