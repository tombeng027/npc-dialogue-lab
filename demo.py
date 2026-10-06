from engine import NPCDialogueEngine
from models import CharacterProfile, DialogueContext


def main() -> None:
    grom = CharacterProfile(
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
                "Collect heatproof stones, recover Grom's missing hammer, then repair "
                "and relight the furnace using both."
            )
        },
        backstory="Former royal guard who retired after a betrothal dispute.",
    )
    context = DialogueContext(
        location="Ironforge Smithy",
        current_event="The town guard is searching for a runaway thief.",
        player_relationship="Stranger",
    )
    engine = NPCDialogueEngine(profile=grom)

    for player_input in (
        "Can you make me a sword, and can I get a discount?",
        "Fine. Is there anything useful I can do around here?",
    ):
        response = engine.speak(player_input, context)
        print(f"{grom.name} [{response.emotion.value}]: {response.dialogue}")
        if response.quest_triggered:
            print(f"Quest event: {response.quest_triggered}")


if __name__ == "__main__":
    main()