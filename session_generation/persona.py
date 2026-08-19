persona_1 = {
    "character":
    "I'm a senior backend engineer at a mid-size SaaS company; my day job is "
    "systems and APIs, so I'm strong on infrastructure, security trade-offs, and "
    "how engineering orgs actually ship — but I'm only hobby-level on consumer "
    "gadgets and semiconductors, and I lean on others for pure policy nuance. I "
    "follow technology news to stay credible with peers and to see where the "
    "industry is heading. I'm influenced by engineering blogs, conference talks, "
    "and skeptical podcasts — I prefer plain language and concrete examples and "
    "get impatient with vague hype.",
}

persona_2 = {
    "character":
    "I'm a corporate strategy analyst at a diversified industrial firm; I spend "
    "most of my time on markets, competitive dynamics, and board-ready "
    "storylines. I'm fluent in business and finance news and comfortable "
    "translating technical stories into dollars-and-cents implications, but I'm "
    "not a technologist — I know enough about cloud, AI, and cybersecurity to ask "
    "sharp questions, not to implement anything. I'm influenced by long-form "
    "business journalism and equity research: skeptical of tidy narratives and "
    "allergic to buzzwords without numbers behind them.",
}

persona_3 = {
    "character":
    "I'm a high school athletic director and former college basketball player; "
    "sports are both my job and my main lens on the world. I'm deep on rules, "
    "coaching culture, injury reporting, and how narratives form around teams and "
    "stars; I'm a casual fan in sports I don't oversee and I don't pretend to be "
    "a stats researcher. I'm influenced by beat writers, national columnists, and "
    "sports radio — I like storytelling with accountability and bristle at hot "
    "takes that ignore what actually happened on the field.",
}

persona_4 = {
    "character":
    "I'm a freelance newsletter writer and part-time community college "
    "instructor; I read widely because my beat is 'whatever readers are asking "
    "about this week,' not because I'm an expert in every domain. I'm strongest "
    "on business and technology stories I can tie to work and money, passably "
    "informed on health and science when I do my homework, and honest that "
    "entertainment and sports are mostly enthusiastic amateur territory for me — "
    "I know enough not to sound foolish, not enough to lecture anyone. I'm "
    "influenced by general-interest podcasts, newspaper features, and teacher "
    "friends — I value clarity over cleverness and I'm comfortable admitting gaps.",
}

PERSONA_BY_ID: dict[str, dict[str, str]] = {
    "persona_1": persona_1,
    "persona_2": persona_2,
    "persona_3": persona_3,
    "persona_4": persona_4,
}


def persona_dict_for_id(persona_id: str) -> dict[str, str]:
    """Resolve a persona label from question→persona allocation to a character dict.

    Raises ``ValueError`` if ``persona_id`` is not a known id (``persona_1`` …
    ``persona_4``). There is no fallback persona.
    """
    pid = persona_id.strip()
    if pid not in PERSONA_BY_ID:
        raise ValueError(
            f"Unknown persona id {persona_id!r}; expected one of "
            f"{sorted(PERSONA_BY_ID)}"
        )
    return PERSONA_BY_ID[pid]

# ---------------------------------------------------------------------------
# Diversity pools — independently sampled per batch request
# ---------------------------------------------------------------------------

SCENARIOS = [
    # Professional / work-oriented (worded for any job: office, school, sports,
    # freelance, public-facing, etc.)
    {
        "description": (
            "The user is drafting a professional message to send or share "
            "(for example with coworkers, staff, parents, clients, vendors, or "
            "the public) and references the article as supporting material."
        ),
        "category": "professional",
    },
    {
        "description": (
            "The user is doing quick research before an obligation where they "
            "will need to explain or decide something — such as a meeting, "
            "class, briefing, interview, game-day or event prep, or a public "
            "appearance — and needs to get up to speed on this topic."
        ),
        "category": "professional",
    },
    {
        "description": (
            "The user is preparing notes, talking points, or remarks for an "
            "upcoming briefing, update, workshop, hearing, or presentation."
        ),
        "category": "professional",
    },
    {
        "description": (
            "The user is writing a short piece for public or semi-public "
            "channels — a post, newsletter blurb, blog draft, flyer, or handout "
            "— inspired by the article."
        ),
        "category": "professional",
    },
    {
        "description": (
            "The user is exploring what this news could mean for their "
            "organization, program, budget, or community, or for people who rely "
            "on them (such as stakeholders, a board, a team, a district, or "
            "sponsors)."
        ),
        "category": "professional",
    },
    {
        "description": (
            "The user is summarizing or breaking down the article for someone "
            "in their professional orbit who asked — a coworker, staff member, "
            "editor, volunteer, or counterpart at another organization."
        ),
        "category": "professional",
    },
    # Casual / personal-life
    {
        "description": (
            "The user is fact-checking something to prove a friend wrong in a "
            "group chat argument."
        ),
        "category": "casual",
    },
    {
        "description": (
            "The user is settling a debate with a partner or family member "
            "about something in the news."
        ),
        "category": "casual",
    },
    {
        "description": (
            "The user is satisfying idle curiosity after scrolling past a "
            "headline on their phone."
        ),
        "category": "casual",
    },
    {
        "description": (
            "The user wants to sound informed about a trending topic before a "
            "dinner party or social event."
        ),
        "category": "casual",
    },
    {
        "description": (
            "The user is explaining a news story to a parent or friend who "
            "asked them about it."
        ),
        "category": "casual",
    },
    {
        "description": (
            "The user is casually exploring a topic they stumbled on — no "
            "deliverable, just thinking out loud."
        ),
        "category": "casual",
    },
    # Quick / transactional
    {
        "description": (
            "The user wants to verify or fact-check a single specific claim "
            "from the article."
        ),
        "category": "transactional",
    },
    {
        "description": (
            "The user needs one quick thing from the assistant and wants to "
            "move on."
        ),
        "category": "transactional",
    },
    {
        "description": (
            "The user is comparing this article's perspective to something "
            "else they read or heard."
        ),
        "category": "transactional",
    },
]

TURN_RANGES = [
    (4, 4),   # minimal / quick exchange
    (4, 6),   # short conversation
    (6, 10),  # longer, more developed session
    (8, 10),  # extended, in-depth session
]

# Aligns with ``TURN_RANGES`` — skew toward shorter sessions (typical chat mix).
TURN_RANGE_WEIGHTS = (0.35, 0.25, 0.25, 0.15)

# Canonical drift level order for weight tuples: tight → moderate → wide.
TOPIC_DRIFT_LEVELS = ("tight", "moderate", "wide")

# Per turn-range bucket: P(tight), P(moderate), P(wide). Shorter sessions skew
# tighter; longer sessions allow more drift.
TOPIC_DRIFT_WEIGHTS_BY_TURN_RANGE: dict[tuple[int, int], tuple[float, float, float]] = {
    (4, 4): (0.88, 0.10, 0.02),
    (4, 6): (0.78, 0.18, 0.04),
    (6, 10): (0.5, 0.4, 0.1),
    (8, 10): (0.45, 0.45, 0.1),
}

EVIDENCE_PLACEMENTS = {
    "early": (
        "The user introduces the evidence in their first or second message."
    ),
    "middle": (
        "The user introduces the evidence around the middle of the "
        "conversation, not in the first turn."
    ),
    "late": (
        "The user introduces the evidence toward the end of the conversation, "
        "after other topics have been discussed."
    ),
}

TOPIC_DRIFTS = {
    "tight": (
        "The conversation stays closely focused on the evidence topic."
    ),
    "moderate": (
        "The conversation includes one or two adjacent tasks or follow-ups "
        "beyond the core evidence topic."
    ),
    "wide": (
        "The conversation wanders across multiple topics or tasks; the "
        "evidence is one thread among others."
    ),
}

EVIDENCE_DENSITIES = {
    "single_message": (
        "The user introduces all evidence details (title, source, and key "
        "information) in a single message."
    ),
    "multi_message": (
        "The user introduces the evidence details gradually, spread across "
        "two or more of their messages rather than all at once."
    ),
}

TRANSACTIONAL_CONSTRAINTS = {
    "turn_ranges": [(4, 4), (4, 6)],
    "evidence_placements": ["early"],
    "topic_drifts": ["tight", "moderate"],
    "evidence_densities": ["single_message", "multi_message"],
}

# Aligns with ``turn_ranges`` / ``topic_drifts`` in ``TRANSACTIONAL_CONSTRAINTS``.
TRANSACTIONAL_TURN_RANGE_WEIGHTS = (0.55, 0.45)
TRANSACTIONAL_TOPIC_DRIFT_WEIGHTS = (0.78, 0.22)


# ---------------------------------------------------------------------------
# Per-persona diversity profiles
# ---------------------------------------------------------------------------
# Each profile overrides the global weights above. The same profile is used for
# both on-topic and off-topic evidence (no structural confound between them).
# Keys mirror the global weight tuples: scenario_category_weights maps category
# labels to selection probability; the other tuples align with their respective
# global pools (TURN_RANGES, TOPIC_DRIFT_LEVELS, etc.).

PERSONA_DIVERSITY_PROFILES: dict[str, dict] = {
    "persona_1": {
        # Tech engineer: skews professional, moderate-to-long sessions
        "scenario_category_weights": {"professional": 0.55, "casual": 0.30, "transactional": 0.15},
        "turn_range_weights": (0.15, 0.30, 0.35, 0.20),
        "topic_drift_weights_by_turn_range": {
            (4, 4): (0.80, 0.15, 0.05),
            (4, 6): (0.70, 0.22, 0.08),
            (6, 10): (0.45, 0.42, 0.13),
            (8, 10): (0.40, 0.45, 0.15),
        },
    },
    "persona_2": {
        # Business+Tech analyst: balanced professional/casual, moderate length
        "scenario_category_weights": {"professional": 0.50, "casual": 0.32, "transactional": 0.18},
        "turn_range_weights": (0.20, 0.30, 0.30, 0.20),
        "topic_drift_weights_by_turn_range": {
            (4, 4): (0.82, 0.14, 0.04),
            (4, 6): (0.72, 0.20, 0.08),
            (6, 10): (0.48, 0.40, 0.12),
            (8, 10): (0.42, 0.44, 0.14),
        },
    },
    "persona_3": {
        # Sports AD: skews casual, shorter sessions
        "scenario_category_weights": {"professional": 0.30, "casual": 0.50, "transactional": 0.20},
        "turn_range_weights": (0.30, 0.30, 0.25, 0.15),
        "topic_drift_weights_by_turn_range": {
            (4, 4): (0.75, 0.18, 0.07),
            (4, 6): (0.65, 0.25, 0.10),
            (6, 10): (0.42, 0.42, 0.16),
            (8, 10): (0.38, 0.44, 0.18),
        },
    },
}


# Domain sets for determining on-topic vs off-topic per persona.
PERSONA_DOMAINS: dict[str, frozenset[str]] = {
    "persona_1": frozenset({"technology"}),
    "persona_2": frozenset({"business", "technology"}),
    "persona_3": frozenset({"sports"}),
}


def is_off_topic_for_persona(persona_id: str, evidence_categories: set[str]) -> bool:
    """Return True if none of the evidence categories overlap with the persona's domains."""
    domains = PERSONA_DOMAINS.get(persona_id)
    if domains is None:
        return False
    return not domains.intersection(evidence_categories)


def diversity_profile_for_persona(persona_id: str) -> dict | None:
    """Return the diversity profile for a persona, or None for global defaults."""
    return PERSONA_DIVERSITY_PROFILES.get(persona_id)