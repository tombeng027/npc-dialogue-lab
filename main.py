import hashlib
import logging
import os
import re
from threading import RLock
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine import NPCDialogueEngine, NPCDialogueError
from models import (
    CharacterProfile,
    DialogueContext,
    NPCResponse,
    QuestRequirement,
    WorldAction,
    WorldLocation,
)
from world import WorldState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NPC Dialogue Engine", version="1.0.0")

GROM_PROFILE = CharacterProfile(
    name="Grom",
    age=48,
    gender="Male",
    job="Blacksmith",
    attitude="Grumpy, impatient, values hard work, dislikes chit-chat",
    likes=["Steel", "Cold ale", "Honest trade"],
    dislikes=["Bargaining", "Nobility", "Time-wasters"],
    quest_lines=["fix_the_furnace", "retrieve_mithril_ore"],
    quest_guidance={
        "fix_the_furnace": (
            "The player must collect heatproof stones, find and recover Grom's missing "
                "hammer, then use both to repair and relight the furnace. Use actions "
                "gather_heatproof_stones, recover_grom_hammer, then repair_furnace in that order."
        ),
        "retrieve_mithril_ore": (
            "The player must mine mithril ore at Deepstone Mine, then deliver the ore to "
            "Grom at Ironforge Smithy. Use gather_mithril_ore before deliver_mithril_ore."
        ),
    },
    quest_rewards={"fix_the_furnace": 50, "retrieve_mithril_ore": 30},
    backstory="Former royal guard who retired after a betrothal dispute.",
)
NPC_PROFILES = {"grom": GROM_PROFILE}
_engines: dict[str, NPCDialogueEngine] = {}
_engines_lock = RLock()
world_state = WorldState()


class NPCSummary(BaseModel):
    npc_id: str
    name: str
    job: str
    relationship_score: int


class NPCDetails(BaseModel):
    npc_id: str
    profile: CharacterProfile
    relationship_score: int
    available_quests: list[str]
    triggered_quests: list[str]
    quest_requirements: dict[str, list[QuestRequirement]]
    quest_rewards: dict[str, int]
    conversation_turns: int


class NPCConversationTurn(BaseModel):
    player_input: str
    response: NPCResponse


class NPCConversation(BaseModel):
    npc_id: str
    name: str
    relationship_score: int
    turns: list[NPCConversationTurn]


class NPCChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_input: str
    context: DialogueContext


class LocationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    thumbnail_url: str | None = None


class WorldActionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    mode: Literal["gather", "use", "sell"]
    reward_item_id: str | None = None
    reward_item_name: str | None = None
    reward_quantity: int = Field(default=1, ge=1, le=999)
    consumes: dict[str, int] = Field(default_factory=dict)
    money_reward: int = Field(default=0, ge=0, le=1_000_000)


class InventoryGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=100)
    item_name: str = Field(min_length=1, max_length=100)
    quantity: int = Field(default=1, ge=1, le=999)


class WorldActionResult(BaseModel):
    location: WorldLocation
    action: WorldAction
    inventory: list[dict[str, object]]
    money_earned: int
    wallet_balance: int
    quest_rewards: list[dict[str, object]] = Field(default_factory=list)


class TroubleshootingReport(BaseModel):
    code: str
    message: str
    suggestions: list[str]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_input: str
    context: DialogueContext
    npc_id: str | None = None
    profile: CharacterProfile | None = None

    @model_validator(mode="after")
    def require_npc_identifier_or_profile(self) -> "ChatRequest":
        if (self.npc_id is None) == (self.profile is None):
            raise ValueError("Provide exactly one of npc_id or profile.")
        return self


def _get_engine(request: ChatRequest) -> NPCDialogueEngine:
    if request.npc_id is not None:
        return _get_engine_by_id(request.npc_id)

    profile = request.profile
    if profile is None:
        raise HTTPException(status_code=422, detail="An NPC profile is required.")
    profile_hash = hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()
    cache_key = f"profile:{profile_hash}"

    with _engines_lock:
        if cache_key not in _engines:
            _engines[cache_key] = NPCDialogueEngine(profile=profile)
        return _engines[cache_key]


def _get_engine_by_id(npc_id: str) -> NPCDialogueEngine:
    with _engines_lock:
        profile = NPC_PROFILES.get(npc_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown npc_id.")
        cache_key = f"npc:{npc_id}"
        if cache_key not in _engines:
            _engines[cache_key] = NPCDialogueEngine(profile=profile)
        return _engines[cache_key]


def _check_provider() -> None:
    provider = os.getenv("NPC_LLM_PROVIDER", "ollama").lower()
    if provider not in {"ollama", "openai"}:
        raise HTTPException(status_code=503, detail="Dialogue provider is not supported.")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="Dialogue service is not configured.")


def _troubleshooting_report(error: NPCDialogueError) -> TroubleshootingReport:
    message = str(error)
    if error.code != "dialogue_error":
        return TroubleshootingReport(
            code=error.code,
            message=message,
            suggestions=error.suggestions,
        )

    if message.startswith("No world action can complete quest "):
        quest_id = message.partition("'")[2].partition("'")[0]
        return TroubleshootingReport(
            code="missing_quest_action",
            message=f"Quest '{quest_id}' has no configured world action that completes it.",
            suggestions=[
                "Open World & Inventory and select a location where the final quest action can happen.",
                "Add an action with type 'Use items / perform task' and list its required inventory as item_id:quantity.",
                "For required items, add gather actions that reward those exact item IDs, then retry the conversation.",
                "Update the NPC's quest requirements guidance to name the configured actions and item IDs.",
            ],
        )

    if message.startswith("No world action can gather required item "):
        item_id = message.partition("'")[2].partition("'")[0]
        return TroubleshootingReport(
            code="missing_gather_action",
            message=f"A quest action requires '{item_id}', but no world action produces that item.",
            suggestions=[
                "Add a location action with type 'Gather an item'.",
                f"Set its reward item ID to exactly `{item_id}` and provide a display name and quantity.",
                "Retry the NPC conversation after the gather action appears in the map.",
            ],
        )

    if "before collecting" in message or "missing required items" in message:
        return TroubleshootingReport(
            code="quest_action_dependencies",
            message=message,
            suggestions=[
                "Add a gather action for each missing item, using the exact item IDs listed by the use action.",
                "Ensure the quest plan collects those items before its use action consumes them.",
                "Review the NPC's quest guidance so it refers to the configured locations and actions.",
            ],
        )

    return TroubleshootingReport(
        code="quest_plan_invalid",
        message=message,
        suggestions=[
            "Check that the NPC quest ID exists in its Quest lines field.",
            "Ensure every objective maps to a real map action and that required items can be gathered.",
            "Review the NPC's quest requirements guidance, then retry the conversation.",
        ],
    )


def _speak(engine: NPCDialogueEngine, player_input: str, context: DialogueContext) -> NPCResponse:
    try:
        return engine.speak(
            player_input,
            context,
            world_actions=world_state.actions(),
            inventory=world_state.inventory_counts(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NPCDialogueError as exc:
        logger.exception("NPC dialogue generation failed")
        raise HTTPException(
            status_code=502,
            detail=_troubleshooting_report(exc).model_dump(),
        ) from exc


@app.get("/", include_in_schema=False)
def workbench() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/world", response_model=list[WorldLocation])
def get_world() -> list[WorldLocation]:
    return world_state.locations()


@app.post("/world/locations", response_model=WorldLocation, status_code=201)
def create_location(request: LocationCreateRequest) -> WorldLocation:
    try:
        return world_state.create_location(
            request.name,
            request.description,
            request.thumbnail_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.put("/world/locations/{location_id}", response_model=WorldLocation)
def update_location(location_id: str, request: LocationCreateRequest) -> WorldLocation:
    try:
        return world_state.update_location(
            location_id,
            request.name,
            request.description,
            request.thumbnail_url,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/world/locations/{location_id}/actions", response_model=WorldLocation, status_code=201)
def create_world_action(location_id: str, request: WorldActionCreateRequest) -> WorldLocation:
    try:
        location = world_state.location(location_id)
        action_slug = re.sub(r"[^a-z0-9]+", "_", request.name.lower()).strip("_")
        action = WorldAction(
            action_id=f"{location_id}_{action_slug}",
            location_id=location_id,
            location_name=location.name,
            name=request.name.strip(),
            description=request.description.strip(),
            mode=request.mode,
            reward_item_id=request.reward_item_id,
            reward_item_name=request.reward_item_name,
            reward_quantity=request.reward_quantity,
            consumes=request.consumes,
            money_reward=request.money_reward,
        )
        return world_state.create_action(location_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.put("/world/locations/{location_id}/actions/{action_id}", response_model=WorldLocation)
def update_world_action(
    location_id: str,
    action_id: str,
    request: WorldActionCreateRequest,
) -> WorldLocation:
    try:
        location = world_state.location(location_id)
        action = WorldAction(
            action_id=action_id,
            location_id=location_id,
            location_name=location.name,
            name=request.name.strip(),
            description=request.description.strip(),
            mode=request.mode,
            reward_item_id=request.reward_item_id,
            reward_item_name=request.reward_item_name,
            reward_quantity=request.reward_quantity,
            consumes=request.consumes,
            money_reward=request.money_reward,
        )
        return world_state.update_action(location_id, action_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/inventory", response_model=list[dict[str, object]])
def get_inventory() -> list[dict[str, object]]:
    return world_state.inventory()


@app.get("/wallet", response_model=dict[str, int])
def get_wallet() -> dict[str, int]:
    return {"balance": world_state.money()}


@app.post("/inventory/items", response_model=list[dict[str, object]])
def grant_inventory_item(request: InventoryGrantRequest) -> list[dict[str, object]]:
    try:
        updated = world_state.grant_item(request.item_id, request.item_name.strip(), request.quantity)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    inventory = world_state.inventory_counts()
    for npc_id in NPC_PROFILES:
        engine = _engines.get(f"npc:{npc_id}")
        if engine:
            for quest_id, amount in engine.apply_inventory(inventory):
                world_state.add_money(amount)
    return updated


@app.post("/world/actions/{action_id}/perform", response_model=WorldActionResult)
def perform_world_action(action_id: str) -> WorldActionResult:
    try:
        location, action, inventory, money_earned = world_state.perform_action(action_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    inventory_counts = world_state.inventory_counts()
    quest_rewards: list[dict[str, object]] = []
    for npc_id in NPC_PROFILES:
        engine = _engines.get(f"npc:{npc_id}")
        if engine:
            completed_rewards = engine.apply_world_action(action, inventory_counts)
            for quest_id, amount in completed_rewards:
                balance = world_state.add_money(amount)
                money_earned += amount
                quest_rewards.append({"npc_id": npc_id, "quest_id": quest_id, "amount": amount, "wallet_balance": balance})
    return WorldActionResult(
        location=location,
        action=action,
        inventory=inventory,
        money_earned=money_earned,
        wallet_balance=world_state.money(),
        quest_rewards=quest_rewards,
    )


@app.post("/npcs", response_model=NPCSummary, status_code=201)
def create_npc(profile: CharacterProfile) -> NPCSummary:
    _check_provider()
    npc_id = f"npc_{uuid4().hex[:12]}"
    with _engines_lock:
        NPC_PROFILES[npc_id] = profile
    return NPCSummary(
        npc_id=npc_id,
        name=profile.name,
        job=profile.job,
        relationship_score=50,
    )


@app.put("/npcs/{npc_id}", response_model=NPCSummary)
def update_npc(npc_id: str, profile: CharacterProfile) -> NPCSummary:
    with _engines_lock:
        if npc_id not in NPC_PROFILES:
            raise HTTPException(status_code=404, detail="Unknown npc_id.")

        engine = _engines.get(f"npc:{npc_id}")
        started_quests = set(engine.quest_events) if engine else set()
        removed_quests = started_quests.difference(profile.quest_lines)
        if removed_quests:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot remove quest lines that already have progress: "
                    f"{', '.join(sorted(removed_quests))}."
                ),
            )

        NPC_PROFILES[npc_id] = profile
        if engine:
            engine.profile = profile
        relationship_score = engine.relationship_score if engine else 50

    return NPCSummary(
        npc_id=npc_id,
        name=profile.name,
        job=profile.job,
        relationship_score=relationship_score,
    )


@app.put("/npcs/{npc_id}", response_model=NPCSummary)
def update_npc(npc_id: str, profile: CharacterProfile) -> NPCSummary:
    with _engines_lock:
        if npc_id not in NPC_PROFILES:
            raise HTTPException(status_code=404, detail="Unknown npc_id.")
        engine = _engines.get(f"npc:{npc_id}")
        started_quests = set(engine.quest_events) if engine else set()
        removed_quests = started_quests.difference(profile.quest_lines)
        if removed_quests:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot remove quest lines that already have progress: "
                    f"{', '.join(sorted(removed_quests))}."
                ),
            )
        NPC_PROFILES[npc_id] = profile
        if engine:
            engine.profile = profile
        relationship_score = engine.relationship_score if engine else 50

    return NPCSummary(
        npc_id=npc_id,
        name=profile.name,
        job=profile.job,
        relationship_score=relationship_score,
    )


@app.get("/npcs", response_model=list[NPCSummary])
def list_npcs() -> list[NPCSummary]:
    with _engines_lock:
        return [
            NPCSummary(
                npc_id=npc_id,
                name=profile.name,
                job=profile.job,
                relationship_score=_engines[f"npc:{npc_id}"].relationship_score
                if f"npc:{npc_id}" in _engines
                else 50,
            )
            for npc_id, profile in NPC_PROFILES.items()
        ]


@app.get("/npcs/{npc_id}", response_model=NPCDetails)
def get_npc_details(npc_id: str) -> NPCDetails:
    engine = _get_engine_by_id(npc_id)
    with _engines_lock:
        profile = NPC_PROFILES.get(npc_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown npc_id.")
        return NPCDetails(
            npc_id=npc_id,
            profile=profile,
            relationship_score=engine.relationship_score,
            available_quests=profile.quest_lines,
            triggered_quests=engine.quest_events,
            quest_requirements=engine.quest_requirements,
            quest_rewards=profile.quest_rewards,
            conversation_turns=engine.turn_count,
        )


@app.get("/npcs/{npc_id}/conversation", response_model=NPCConversation)
def get_conversation(npc_id: str) -> NPCConversation:
    _check_provider()
    engine = _get_engine_by_id(npc_id)
    turns: list[NPCConversationTurn] = []
    history = engine.history
    for index in range(0, len(history) - 1, 2):
        player_message, npc_message = history[index : index + 2]
        if player_message["role"] != "user" or npc_message["role"] != "assistant":
            continue
        turns.append(
            NPCConversationTurn(
                player_input=player_message["content"],
                response=NPCResponse.model_validate_json(npc_message["content"]),
            )
        )
    profile = NPC_PROFILES[npc_id]
    return NPCConversation(
        npc_id=npc_id,
        name=profile.name,
        relationship_score=engine.relationship_score,
        turns=turns,
    )


@app.delete("/npcs/{npc_id}", status_code=204)
def delete_npc(npc_id: str) -> Response:
    with _engines_lock:
        if npc_id not in NPC_PROFILES:
            raise HTTPException(status_code=404, detail="Unknown npc_id.")
        del NPC_PROFILES[npc_id]
        _engines.pop(f"npc:{npc_id}", None)
    return Response(status_code=204)


@app.post("/npcs/{npc_id}/chat", response_model=NPCResponse)
def chat_with_npc(npc_id: str, request: NPCChatRequest) -> NPCResponse:
    _check_provider()
    engine = _get_engine(
        ChatRequest(
            player_input=request.player_input,
            context=request.context,
            npc_id=npc_id,
        )
    )
    return _speak(engine, request.player_input, request.context)


@app.post("/chat", response_model=NPCResponse)
def chat(request: ChatRequest) -> NPCResponse:
    _check_provider()
    engine = _get_engine(request)
    return _speak(engine, request.player_input, request.context)