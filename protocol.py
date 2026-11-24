"""A2A (Agent2Agent) protocol data model - JSON-RPC 2.0 binding.

Implements the v0.3.0 wire format (the most widely deployed release) and accepts the v1.0
JSON-RPC method names as aliases (SendMessage, GetTask, CancelTask). Spec:
https://a2a-protocol.org/latest/specification/
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "0.3.0"
A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"
A2UI_MIME = "application/json+a2ui"

TaskState = Literal["submitted", "working", "input-required", "completed", "canceled", "failed", "rejected",
                    "auth-required", "unknown"]


def _id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    inputModes: list[str] = Field(default_factory=lambda: ["application/json"])
    outputModes: list[str] = Field(default_factory=lambda: ["application/json"])


class AgentExtension(BaseModel):
    uri: str
    description: str = ""
    required: bool = False
    params: dict[str, Any] | None = None


class AgentCapabilities(BaseModel):
    streaming: bool = True
    pushNotifications: bool = False
    stateTransitionHistory: bool = True
    extensions: list[AgentExtension] = Field(default_factory=list)


class AgentProvider(BaseModel):
    organization: str
    url: str


class AgentCard(BaseModel):
    protocolVersion: str = PROTOCOL_VERSION
    name: str
    description: str
    url: str
    preferredTransport: str = "JSONRPC"
    version: str = "1.0.0"
    provider: AgentProvider | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["application/json", "text/plain"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["application/json"])
    skills: list[AgentSkill]
    securitySchemes: dict[str, Any] = Field(default_factory=lambda: {
        "bearer": {"type": "http", "scheme": "bearer", "description": "Service token (mTLS in production)"}})
    security: list[dict[str, list[str]]] = Field(default_factory=list)


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


Part = TextPart | DataPart


class Message(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: list[Part]
    messageId: str = Field(default_factory=_id)
    contextId: str | None = None
    taskId: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class TaskStatus(BaseModel):
    state: TaskState
    message: Message | None = None
    timestamp: str = Field(default_factory=now_iso)


class Artifact(BaseModel):
    artifactId: str = Field(default_factory=_id)
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None


class Task(BaseModel):
    kind: Literal["task"] = "task"
    id: str = Field(default_factory=_id)
    contextId: str = Field(default_factory=_id)
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


# JSON-RPC error codes (A2A-specific + standard)
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

METHOD_ALIASES = {"SendMessage": "message/send", "SendStreamingMessage": "message/stream",
                  "GetTask": "tasks/get", "CancelTask": "tasks/cancel"}
