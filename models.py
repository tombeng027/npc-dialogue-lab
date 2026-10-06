from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Emotion(str, Enum):
    ANGRY = "angry"
    HAPPY = "happy"
    NERVOUS = "nervous"
    NEUTRAL = "neutral"
    SUSPICIOUS = "suspicious"

class CharacterProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int
    gender: str
    job: str
    attitude: str
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    quest_lines: list[str] = Field(default_factory=list)
    quest_guidance: dict[str, str] = Field(default_factory=dict)
    quest_rewards: dict[str, int] = Field(default_factory=dict)
    backstory: str = ""

    @field_validator("quest_rewards")
    @classmethod
    def validate_quest_rewards(cls, rewards: dict[str, int]) -> dict[str, int]:
        if any(amount < 0 for amount in rewards.values()):
            raise ValueError("Quest coin rewards cannot be negative.")
        return rewards

class DialogueContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    current_event: str
    player_relationship: str
    active_quest_step: str | None = None


class WorldAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    location_id: str
    location_name: str
    name: str
    description: str
    mode: Literal["gather", "use", "sell"]
    reward_item_id: str | None = None
    reward_item_name: str | None = None
    reward_quantity: int = Field(default=1, ge=1, le=999)
    consumes: dict[str, int] = Field(default_factory=dict)
    money_reward: int = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "WorldAction":
        if self.mode == "gather" and not self.reward_item_id:
            raise ValueError("Gather actions must provide a reward_item_id.")
        if self.reward_item_id and not self.reward_item_name:
            raise ValueError("An item reward must include its display name.")
        if any(quantity < 1 for quantity in self.consumes.values()):
            raise ValueError("Consumed item quantities must be positive.")
        if self.mode == "sell" and (not self.consumes or self.money_reward < 1):
            raise ValueError("Sell actions must consume items and pay at least one coin.")
        if self.mode == "gather" and self.money_reward:
            raise ValueError("Gather actions cannot also award coins.")
        return self


class WorldLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location_id: str
    name: str
    description: str
    thumbnail_url: str | None = None
    actions: list[WorldAction] = Field(default_factory=list)


class QuestStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    objective_type: Literal["collect_item", "perform_action"]
    action_id: str
    item_id: str | None = None
    quantity: int = Field(default=1, ge=1, le=999)


class QuestRequirement(QuestStepDraft):
    step_id: str
    location_id: str
    location_name: str
    completed: bool = False
    progress: int = Field(default=0, ge=0)


class NPCResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialogue: str = Field(description="The spoken dialogue line matching the NPC's persona.")
    emotion: Emotion = Field(description="Emotion tag for the line of dialogue.")
    quest_triggered: str | None = Field(
        default=None,
        description="Quest ID if this line offers or progresses a quest.",
    )
    relationship_delta: int = Field(
        ge=-5,
        le=5,
        description="Change to the NPC's relationship score caused by the player's line.",
    )
    relationship_score: int = Field(
        ge=0,
        le=100,
        description="Current relationship score after this conversation turn.",
    )
    quest_requirements: list[QuestRequirement] = Field(
        default_factory=list,
        description="Generated quest steps when this response triggers a quest.",
    )


class NPCUtterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialogue: str = Field(description="The spoken dialogue line matching the NPC's persona.")
    emotion: Emotion = Field(description="Emotion tag for the line of dialogue.")
    quest_triggered: str | None = Field(
        default=None,
        description="Quest ID if this line offers or progresses a quest.",
    )
    relationship_delta: int = Field(
        ge=-5,
        le=5,
        description="How the player's line affects this NPC's relationship score.",
    )
    quest_requirements: list[QuestStepDraft] = Field(
        default_factory=list,
        description=(
            "When quest_triggered is set, return 2-5 ordered, concrete, player-"
            "completable steps. Otherwise return an empty list."
        ),
    )