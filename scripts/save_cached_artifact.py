"""Extract a Sybilion artifact from a persisted MCP tool-result file and write
it into the forecast cache.

The MCP tool returns large artifacts wrapped in markdown (```json ... ```).
When the result is too big for the transcript it is persisted to a JSON file
on disk; this script pulls the artifact JSON back out and stores it cleanly
under cache/<job_id>/<name>.json.

Run:  uv run python scripts/save_cached_artifact.py <persisted_result.json> <output_path.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def extract_artifact_json(persisted_path: Path) -> dict | list:
    raw = json.loads(persisted_path.read_text())
    # The persisted result is a list of content blocks; find the text block.
    text = ""
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                break
    else:
        text = str(raw)

    start = text.find("```json")
    if start == -1:
        raise ValueError("no ```json fence found in persisted result")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    payload = text[start:end].strip()
    return json.loads(payload)


def main() -> None:
    persisted_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    artifact = extract_artifact_json(persisted_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
