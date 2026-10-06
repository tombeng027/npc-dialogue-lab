import re
from threading import RLock

from models import WorldAction, WorldLocation


class WorldState:
    def __init__(self) -> None:
        self._locations: dict[str, WorldLocation] = {}
        self._inventory: dict[str, dict[str, object]] = {}
        self._money = 0
        self._lock = RLock()
        self._seed_world()

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        if not slug:
            raise ValueError("Names must include at least one letter or number.")
        return slug

    def _seed_world(self) -> None:
        mine = WorldLocation(
            location_id="deepstone_mine",
            name="Deepstone Mine",
            description="A narrow mine cut into the ridge, rich in furnace-grade stone.",
            actions=[
                WorldAction(
                    action_id="gather_heatproof_stones",
                    location_id="deepstone_mine",
                    location_name="Deepstone Mine",
                    name="Gather heatproof stones",
                    description="Mine a bundle of stones suitable for a blacksmith's furnace.",
                    mode="gather",
                    reward_item_id="heatproof_stone",
                    reward_item_name="Heatproof stone",
                    reward_quantity=3,
                ),
                WorldAction(
                    action_id="gather_mithril_ore",
                    location_id="deepstone_mine",
                    location_name="Deepstone Mine",
                    name="Mine mithril ore",
                    description="Extract a small bundle of mithril ore from the deep seam.",
                    mode="gather",
                    reward_item_id="mithril_ore",
                    reward_item_name="Mithril ore",
                    reward_quantity=2,
                ),
            ],
        )
        watchtower = WorldLocation(
            location_id="old_watchtower",
            name="Old Watchtower",
            description="A weathered guard post overlooking the road to Ironforge.",
            actions=[
                WorldAction(
                    action_id="recover_grom_hammer",
                    location_id="old_watchtower",
                    location_name="Old Watchtower",
                    name="Search for Grom's hammer",
                    description="Search the abandoned watchtower for the blacksmith's hammer.",
                    mode="gather",
                    reward_item_id="grom_hammer",
                    reward_item_name="Grom's hammer",
                    reward_quantity=1,
                )
            ],
        )
        smithy = WorldLocation(
            location_id="ironforge_smithy",
            name="Ironforge Smithy",
            description="Grom's busy workshop, where the cold furnace waits to be repaired.",
            actions=[
                WorldAction(
                    action_id="repair_furnace",
                    location_id="ironforge_smithy",
                    location_name="Ironforge Smithy",
                    name="Repair and relight the furnace",
                    description="Fit the furnace stones and relight the forge.",
                    mode="use",
                    consumes={"heatproof_stone": 3, "grom_hammer": 1},
                ),
                WorldAction(
                    action_id="deliver_mithril_ore",
                    location_id="ironforge_smithy",
                    location_name="Ironforge Smithy",
                    name="Deliver mithril ore to Grom",
                    description="Hand over the mithril ore so Grom can prepare it for forging.",
                    mode="use",
                    consumes={"mithril_ore": 2},
                ),
            ],
        )
        self._locations = {
            mine.location_id: mine,
            watchtower.location_id: watchtower,
            smithy.location_id: smithy,
        }

    def locations(self) -> list[WorldLocation]:
        with self._lock:
            return [location.model_copy(deep=True) for location in self._locations.values()]

    def location(self, location_id: str) -> WorldLocation:
        with self._lock:
            location = self._locations.get(location_id)
            if location is None:
                raise KeyError("Unknown location.")
            return location.model_copy(deep=True)

    def actions(self) -> list[WorldAction]:
        with self._lock:
            return [
                action.model_copy(deep=True)
                for location in self._locations.values()
                for action in location.actions
            ]

    def create_location(
        self,
        name: str,
        description: str,
        thumbnail_url: str | None = None,
    ) -> WorldLocation:
        location_id = self._slug(name)
        with self._lock:
            if location_id in self._locations:
                raise ValueError("A location with that name already exists.")
            location = WorldLocation(
                location_id=location_id,
                name=name.strip(),
                description=description.strip(),
                thumbnail_url=thumbnail_url.strip() if thumbnail_url and thumbnail_url.strip() else None,
            )
            self._locations[location_id] = location
            return location.model_copy(deep=True)

    def update_location(
        self,
        location_id: str,
        name: str,
        description: str,
        thumbnail_url: str | None,
    ) -> WorldLocation:
        with self._lock:
            location = self._locations.get(location_id)
            if location is None:
                raise KeyError("Unknown location.")
            location.name = name.strip()
            location.description = description.strip()
            location.thumbnail_url = thumbnail_url.strip() if thumbnail_url and thumbnail_url.strip() else None
            for action in location.actions:
                action.location_name = location.name
            return location.model_copy(deep=True)

    def create_action(self, location_id: str, action: WorldAction) -> WorldLocation:
        with self._lock:
            location = self._locations.get(location_id)
            if location is None:
                raise KeyError("Unknown location.")
            if any(existing.action_id == action.action_id for item in self._locations.values() for existing in item.actions):
                raise ValueError("An action with that name already exists.")
            action = action.model_copy(update={"location_id": location_id, "location_name": location.name})
            location.actions.append(action)
            return location.model_copy(deep=True)

    def update_action(self, location_id: str, action_id: str, action: WorldAction) -> WorldLocation:
        with self._lock:
            location = self._locations.get(location_id)
            if location is None:
                raise KeyError("Unknown location.")
            for index, existing in enumerate(location.actions):
                if existing.action_id == action_id:
                    location.actions[index] = action.model_copy(
                        update={
                            "action_id": action_id,
                            "location_id": location_id,
                            "location_name": location.name,
                        }
                    )
                    return location.model_copy(deep=True)
            raise KeyError("Unknown world action.")

    def inventory(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {"item_id": item_id, **item.copy()}
                for item_id, item in sorted(self._inventory.items())
                if item["quantity"] > 0
            ]

    def inventory_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                item_id: int(item["quantity"])
                for item_id, item in self._inventory.items()
                if int(item["quantity"]) > 0
            }

    def money(self) -> int:
        with self._lock:
            return self._money

    def add_money(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("Money amount cannot be negative.")
        with self._lock:
            self._money += amount
            return self._money

    def grant_item(self, item_id: str, item_name: str, quantity: int) -> list[dict[str, object]]:
        if quantity < 1:
            raise ValueError("Quantity must be at least 1.")
        with self._lock:
            current = self._inventory.get(item_id)
            if current and current["name"] != item_name:
                raise ValueError("That item ID is already registered with a different name.")
            if not current:
                self._inventory[item_id] = {"name": item_name, "quantity": 0}
            self._inventory[item_id]["quantity"] = int(self._inventory[item_id]["quantity"]) + quantity
            return self.inventory()

    def perform_action(
        self,
        action_id: str,
    ) -> tuple[WorldLocation, WorldAction, list[dict[str, object]], int]:
        with self._lock:
            location = next(
                (item for item in self._locations.values() if any(action.action_id == action_id for action in item.actions)),
                None,
            )
            if location is None:
                raise KeyError("Unknown world action.")
            action = next(action for action in location.actions if action.action_id == action_id)

            missing = {
                item_id: quantity - int(self._inventory.get(item_id, {"quantity": 0})["quantity"])
                for item_id, quantity in action.consumes.items()
                if int(self._inventory.get(item_id, {"quantity": 0})["quantity"]) < quantity
            }
            if missing:
                details = ", ".join(f"{item_id}: need {quantity} more" for item_id, quantity in missing.items())
                raise ValueError(f"Not enough inventory to perform this action ({details}).")

            money_earned = action.money_reward
            if action.reward_item_id and action.reward_item_name:
                current_reward = self._inventory.get(action.reward_item_id)
                if current_reward and current_reward["name"] != action.reward_item_name:
                    raise ValueError("The action reward conflicts with an existing inventory item.")

            for item_id, quantity in action.consumes.items():
                self._inventory[item_id]["quantity"] = int(self._inventory[item_id]["quantity"]) - quantity
            if action.reward_item_id and action.reward_item_name:
                current = self._inventory.get(action.reward_item_id)
                if not current:
                    self._inventory[action.reward_item_id] = {"name": action.reward_item_name, "quantity": 0}
                self._inventory[action.reward_item_id]["quantity"] = (
                    int(self._inventory[action.reward_item_id]["quantity"]) + action.reward_quantity
                )
            self._money += money_earned

            return (
                location.model_copy(deep=True),
                action.model_copy(deep=True),
                self.inventory(),
                money_earned,
            )
