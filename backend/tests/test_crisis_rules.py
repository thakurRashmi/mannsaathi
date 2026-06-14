"""
Tests for the deterministic crisis-detection rules.

These are the SAFETY FLOOR of the system. Failures here mean a real crisis
could be missed regardless of how the LLM behaves. Treat them with the
same seriousness you'd treat tests on a payments engine.

Convention:
  - `CRISIS_*` cases must all return is_crisis=True
  - `SAFE_*`   cases must all return is_crisis=False
  - When you add a new pattern to crisis_rules.py, add at least one
    case here for it.
"""
import pytest

from app.agents.crisis_rules import check_rules


# ---- Cases that MUST be flagged (zero false-negatives target) -------------

CRISIS_DIRECT = [
    "I want to die",
    "I wanna die",
    "I want to kill myself",
    "I'm going to commit suicide",
    "I'm thinking about suicide",
    "I'm thinking of killing myself",
    "I want to end my life",
    "I want to end it all",
    "I'm going to end it all",
    "There's no reason to live",
    "I don't want to live anymore",
    "I don't want to be here anymore",
    "Everyone would be better off without me",
    "The world would be better without me",
    "I'm better off dead",
    "I'm going to hang myself",
    "I'm going to jump off a bridge",
    "I'm planning to overdose",
    "I'm suicidal",
]

CRISIS_SELF_HARM = [
    "I've been cutting myself",
    "I want to hurt myself",
    "I keep harming myself",
    "I burn myself when I'm overwhelmed",
    "I've started self-harming again",
]

CRISIS_ABUSE = [
    "My husband hits me every day",
    "She beats me",
    "I'm being abused at home",
    "Someone is hurting me right now",
    "I'm being raped",
]

CRISIS_VIOLENCE = [
    "I'm going to kill him",
    "I'm going to hurt them tomorrow",
    "I'm going to shoot someone",
]

# Mild punctuation / casing variants that MUST still match.
CRISIS_NORMALIZATION = [
    "I.WANT.TO.DIE",
    "i want to die!!!",
    "I — want — to — die",
    "  I want to die  ",
    "I don't want to live",  # apostrophe normalization
    "i dont want to live",
]


# ---- Cases that MUST NOT be flagged (false-positive guard) -----------------
# These look superficially scary but are figurative or unrelated.

SAFE_FIGURATIVE = [
    "This deadline is killing me",
    "I'm dying of laughter",
    "That movie was so good I died",
    "Work is killing me lately",
    "I could die from embarrassment",
]

SAFE_ABSTRACT_DISCUSSION = [
    "I read an article about suicide prevention",
    "My friend went through depression last year",
    "Mental health support is hard to find",
    "I want to learn about self-harm so I can help my brother",
]

SAFE_NORMAL_DISTRESS = [
    "I'm having a rough week",
    "My boss yelled at me",
    "I feel really lonely lately",
    "I'm anxious about my exams",
    "I had an argument with my partner",
    "I feel like nothing is going right",
]


# ---- Parameterized tests --------------------------------------------------

@pytest.mark.parametrize("text", CRISIS_DIRECT + CRISIS_NORMALIZATION)
def test_direct_suicide_patterns_flagged(text: str) -> None:
    result = check_rules(text)
    assert result.is_crisis, f"FALSE NEGATIVE — crisis missed: {text!r}"
    assert "suicide" in result.matched_categories


@pytest.mark.parametrize("text", CRISIS_SELF_HARM)
def test_self_harm_patterns_flagged(text: str) -> None:
    result = check_rules(text)
    assert result.is_crisis, f"FALSE NEGATIVE — self-harm missed: {text!r}"


@pytest.mark.parametrize("text", CRISIS_ABUSE)
def test_abuse_patterns_flagged(text: str) -> None:
    result = check_rules(text)
    assert result.is_crisis, f"FALSE NEGATIVE — abuse missed: {text!r}"


@pytest.mark.parametrize("text", CRISIS_VIOLENCE)
def test_violence_patterns_flagged(text: str) -> None:
    result = check_rules(text)
    assert result.is_crisis, f"FALSE NEGATIVE — violence missed: {text!r}"


@pytest.mark.parametrize("text", SAFE_FIGURATIVE + SAFE_ABSTRACT_DISCUSSION + SAFE_NORMAL_DISTRESS)
def test_safe_messages_not_flagged(text: str) -> None:
    result = check_rules(text)
    assert not result.is_crisis, (
        f"FALSE POSITIVE — normal message flagged: {text!r} "
        f"(matched: {result.matched_patterns})"
    )
