from session_generation.persona import EVIDENCE_PLACEMENTS, EVIDENCE_DENSITIES

# Session-generation instructions for how the simulated dialogue should read (not
# persona identity — that lives in persona ``character`` strings).
DIALOGUE_STYLE_GUIDANCE = (
    "The user's communication style varies naturally between sessions. They may "
    "be concise and task-driven in one conversation and more exploratory or "
    "conversational in another. The assistant should match the user's energy — "
    "brief when the user is brief, more detailed when the user is asking open-ended "
    "questions. Avoid a rigid or formulaic tone; the exchange should read like a "
    "real person talking to their AI assistant."
)


def render_conversation_naturalism(topic_drift: str) -> str:
    """How much the thread may drift from the evidence (tight / moderate / wide).

    This is the only prompt block for topic drift; ``render_structural_guidelines``
    covers turns, placement, and evidence density.
    """
    if topic_drift == "tight":
        return (
            "- Keep the thread focused on the evidence topic and closely related "
            "questions. Naturalism comes from realistic phrasing, small asides, "
            "and imperfect follow-ups — not from unrelated tasks or topic changes."
        )
    if topic_drift == "moderate":
        return (
            "- The evidence should remain a clear thread, but include one or two "
            "adjacent follow-ups or minor side questions that fit the user and "
            "scenario. The chat should not feel as if the evidence were the only "
            "reason they opened the assistant."
        )
    if topic_drift == "wide":
        return (
            "- The interaction should NOT revolve entirely around the evidence. "
            "Include multiple adjacent tasks, follow-up requests, or unrelated "
            "questions that fit the user's profile — the evidence should feel "
            "like one part of a broader work session rather than its sole purpose."
        )
    raise ValueError(f"unknown topic_drift: {topic_drift!r}")


def render_structural_guidelines(
    turn_range: tuple[int, int],
    evidence_placement: str,
    evidence_density: str,
    *,
    is_temporal: bool = False,
) -> str:
    """Build the STRUCTURAL GUIDELINES block (turns, placement, density, USER evidence).

    Topic drift is not included; use ``render_conversation_naturalism`` for that.
    """
    min_turns, max_turns = turn_range
    min_user = min_turns // 2
    max_user = max_turns // 2

    turn_line = (
        f"- Produce a conversation of {min_turns} to {max_turns} turns "
        f"({min_user} to {max_user} user messages, with an equal number of "
        f"assistant responses)."
    )

    placement_line = f"- {EVIDENCE_PLACEMENTS[evidence_placement]}"
    density_line = f"- {EVIDENCE_DENSITIES[evidence_density]}"

    if is_temporal:
        evidence_fields = (
            "title, source, publication time, and key factual details"
        )
        preservation = (
            '- The full factual content of the "Key information", the article '
            "title, the source name, and the publication time must appear in "
            "the USER's messages. Do not omit, paraphrase away, or dilute "
            "critical details (names, numbers, dates, comparisons)."
        )
    else:
        evidence_fields = "title, source, and key factual details"
        preservation = (
            '- The full factual content of the "Key information", the article '
            "title, and the source name must appear in the USER's messages. "
            "Do not omit, paraphrase away, or dilute critical details (names, "
            "numbers, dates, comparisons)."
        )

    static = (
        f"- The user initiates the conversation. The user must introduce the "
        f"evidence ({evidence_fields}) across their messages.\n"
        f"{preservation}"
    )

    return "\n".join([turn_line, placement_line, density_line, static])


# Shared opening (persona + tone). Evidence blocks differ for temporal vs not.
_GENERATION_HEAD = """\
You are generating a realistic past interaction between a user and their AI \
assistant. The assistant is a general-purpose tool for tasks and questions—\
work or everyday—not a companion chatbot. Keep the exchange grounded in \
concrete outcomes.

USER PROFILE:
{persona_summary}

{dialogue_style_guidance}

"""

_EVIDENCE_AND_EMBED = """\
EVIDENCE AND KNOWLEDGE:
The block below is news-article reporting. The assistant has no access to it or any knowledge of the article \
except what the user says in this conversation. Only the user may introduce the \
title, source, {key_fields_intro}and key facts listed—the assistant must not \
state or assume them first. General domain knowledge is fine; this specific \
reporting is not. After the user has supplied those details, the assistant may \
answer, analyze, or help, but must not be the originator of that factual content.

Incorporate the following evidence as something the user discussed with the \
assistant in the past.

EVIDENCE TO EMBED:
{evidence_lines}
"""

_GENERATION_TAIL = """\
SCENARIO:
The scenario below describes the user's situation and motivation. It should \
shape how the user sounds and what they need from the assistant — not be quoted \
or announced explicitly unless a real person would naturally say it that way.

{scenario}

CONVERSATION NATURALISM:
{conversation_naturalism}

STRUCTURAL GUIDELINES:
{structural_guidelines}"""

# Non-temporal: no {published_at}; key_fields_intro is empty ("title, source, and key facts")
GENERATION_SYSTEM = (
    _GENERATION_HEAD
    + _EVIDENCE_AND_EMBED.format(
        key_fields_intro="",
        # Single braces: inserted literally into the template; filled by the final
        # ``.format()`` in ``format_generation_system_prompt`` / batch builder.
        evidence_lines=(
            "- Topic: {title}\n"
            "- Source: {source}\n"
            "- Key information: {fact}"
        ),
    )
    + _GENERATION_TAIL
)

# Temporal: publication time in the rule and in the evidence block
GENERATION_SYSTEM_TEMPORAL = (
    _GENERATION_HEAD
    + _EVIDENCE_AND_EMBED.format(
        key_fields_intro="publication time, ",
        evidence_lines=(
            "- Topic: {title}\n"
            "- Source: {source}\n"
            "- Published at: {published_at}\n"
            "- Key information: {fact}"
        ),
    )
    + _GENERATION_TAIL
)

GENERATION_HUMAN_VARIANTS = [
    "Generate the conversation session now.",
    "Begin the conversation.",
    "Produce the session. Remember to vary the structure and tone from a typical pattern.",
    "Write the conversation. Make it feel like a real, lived interaction — not a template.",
    "Generate the session now. Aim for naturalism over polish.",
    "Start the session. Keep it grounded and believable.",
    "Write the interaction now. Prioritize how a real person would actually talk to their assistant.",
    "Go ahead — generate the conversation.",
    "Create the session. Focus on making the user sound like a real human.",
    "Produce the conversation now. Let the user's personality come through naturally.",
]


# ---------------------------------------------------------------------------
# Off-topic context block — appended after SCENARIO when the evidence is
# out-of-domain for the target persona. Provides framing so the model can
# reconcile any scenario (including professional) with off-topic evidence
# without needing structural parameter changes.
# ---------------------------------------------------------------------------

TOPIC_CONTEXT_OFF_TOPIC = (
    "\n\nTOPIC CONTEXT:\n"
    "This topic is outside the user's primary domain. They are engaging with it "
    "as a general-interest reader — through curiosity, social context, or "
    "incidental exposure. If the scenario is work-related, the connection to this "
    "topic should be indirect (e.g., a colleague mentioned it, it came up in "
    "passing, or they encountered it while looking for something else). The user "
    "should not sound like a domain expert on this topic. "
)
