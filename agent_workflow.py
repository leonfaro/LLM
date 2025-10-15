from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.items import TResponseInputItem
from agents.memory.session import SessionABC
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

__all__ = ["WorkflowInput", "run_workflow"]


def _default_key_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "openai_api_key.txt",
        script_dir.parent / "openai_api_key.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _ensure_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    key_path = _default_key_path()
    if not key_path.exists():
        return
    try:
        key = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if key:
        os.environ["OPENAI_API_KEY"] = key


_ensure_api_key()


def _load_instruction(filename: str) -> str:
    path = Path(__file__).resolve().parent / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Instruction file not found: {filename}") from exc


agent1 = Agent(
    name="Agent1",
    instructions=_load_instruction("agent1_instructions.txt"),
    model="gpt-5",
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(
            effort="minimal",
            summary="detailed",
        ),
    ),
)

agent2 = Agent(
    name="Agent2",
    instructions=_load_instruction("agent2_instructions.txt"),
    model="gpt-5",
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(
            effort="low",
            summary="detailed",
        ),
    ),
)


def _clean_input_item(item: TResponseInputItem) -> TResponseInputItem:
    if hasattr(item, "to_input_item"):
        item = item.to_input_item()  # type: ignore[assignment]
    elif hasattr(item, "model_dump"):
        item = item.model_dump(exclude_unset=True)  # type: ignore[assignment]
    elif isinstance(item, dict):
        item = dict(item)

    if isinstance(item, dict):
        return {k: _clean_input_item(v) for k, v in item.items() if k != "status"}
    if isinstance(item, list):
        return [_clean_input_item(v) for v in item]  # type: ignore[return-value]
    return item


def _append_session_items(
    history_items: List[TResponseInputItem],
    new_items: List[TResponseInputItem],
) -> List[TResponseInputItem]:
    cleaned_history = [_clean_input_item(item) for item in history_items]
    cleaned_new = [_clean_input_item(item) for item in new_items]
    return cleaned_history + cleaned_new



class WorkflowInput(BaseModel):
    input_as_text: str
    screenshots: List[str] = []
    cv_text: Optional[str] = None
    prior_interview_notes: Optional[str] = None
    constraints_or_prefs: Optional[str] = None


async def run_workflow(
    workflow_input: WorkflowInput,
    session: Optional[SessionABC] = None,
    on_stage: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    workflow = workflow_input.model_dump()
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": workflow["input_as_text"]}]

    cv = (workflow.get("cv_text") or "").strip()
    if cv:
        content.append({"type": "input_text", "text": f"cv_text:\n{cv}"})

    notes = (workflow.get("prior_interview_notes") or "").strip()
    if notes:
        content.append({"type": "input_text", "text": f"prior_interview_notes:\n{notes}"})

    cons = (workflow.get("constraints_or_prefs") or "").strip()
    if cons:
        content.append({"type": "input_text", "text": f"constraints_or_prefs:\n{cons}"})

    for path in workflow.get("screenshots", []):
        if not path:
            continue
        try:
            mime = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
        except Exception:
            continue

    conversation_items: List[TResponseInputItem] = [{"role": "user", "content": content}]

    agent1_result = await Runner.run(
        agent1,
        input=conversation_items,
        session=session,
        run_config=RunConfig(
            trace_metadata={"__trace_source__": "agent-builder", "workflow_id": "wf_codex_cli"},
            session_input_callback=_append_session_items,
        ),
    )

    agent1_output = agent1_result.final_output_as(str)
    if on_stage and agent1_output:
        try:
            on_stage("Agent1", agent1_output)
        except Exception:
            pass

    # Build Agent2 input: identical to Agent1 plus Agent1 opener
    agent2_content = list(content)
    if agent1_output:
        agent2_content.append({"type": "input_text", "text": f"Agent1 opener:\n{agent1_output}"})
    agent2_items = [{"role": "user", "content": agent2_content}]
    agent2_result = await Runner.run(
        agent2,
        input=agent2_items,
        session=session,
        run_config=RunConfig(
            trace_metadata={"__trace_source__": "agent-builder", "workflow_id": "wf_codex_cli"},
            session_input_callback=_append_session_items,
        ),
    )

    agent2_output = agent2_result.final_output_as(str)
    if on_stage and agent2_output:
        try:
            on_stage("Agent2", agent2_output)
        except Exception:
            pass
    # Return separate fields; keep output_text as Agent2 for backward compatibility
    return {
        "output_text": (agent2_output or "").strip(),
        "agent1_output": (agent1_output or "").strip(),
        "agent2_output": (agent2_output or "").strip(),
    }
