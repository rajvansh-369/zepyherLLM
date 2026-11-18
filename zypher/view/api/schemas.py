"""Request bodies. Responses are plain dicts in the OpenAI shapes."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from zypher import config
from zypher.model.memory import ABOUT_FIELDS, NOTE_MAX_CHARS


class TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    text: Optional[str] = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[TextPart]]

    def as_dict(self):
        """{"role", "content"} with multi-part content flattened to its text."""

        content = self.content

        if not isinstance(content, str):
            content = "".join(part.text or "" for part in content if part.type == "text")

        return {"role": self.role, "content": content}


class ChatRequest(BaseModel):
    # Unknown OpenAI fields (n, stop, seed, ...) are accepted and ignored, so
    # an existing client does not have to be trimmed down to talk to this.
    model_config = ConfigDict(extra="ignore")

    # Left out, or a name not served here: the model loaded now answers.
    # "general" / "code" or a model id: that model, loaded first if needed
    # (the zy CLI sends "code").
    model: Optional[str] = None
    messages: List[Message] = Field(min_length=1)
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)

    # Runner extensions.
    web: Union[bool, Literal["auto"]] = "auto"
    memory: Optional[bool] = None
    sampling: Optional[Literal["precise", "balanced", "creative"]] = None
    min_tokens: Optional[int] = Field(default=None, ge=1)


class NoteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=NOTE_MAX_CHARS)


class AboutRequest(BaseModel):
    """What the user says about themselves. A field left out is unchanged."""

    name: Optional[str] = Field(default=None, max_length=ABOUT_FIELDS["name"])
    role: Optional[str] = Field(default=None, max_length=ABOUT_FIELDS["role"])
    about: Optional[str] = Field(default=None, max_length=ABOUT_FIELDS["about"])
    style: Optional[str] = Field(default=None, max_length=ABOUT_FIELDS["style"])


class RateRequest(BaseModel):
    rating: Literal["good", "bad"]


class SettingsRequest(BaseModel):
    web: Optional[bool] = None
    memory: Optional[bool] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # The model to load: "general" or "code". Loading takes up to a minute;
    # /health reports "loading" until it is done.
    model: Optional[Literal["general", "code"]] = None
