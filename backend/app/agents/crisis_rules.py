"""
Deterministic crisis-detection rules.

This is the FIRST and most important layer of the safety system. It runs
before any LLM. The point is: even if every LLM in the stack fails
catastrophically (down, hallucinating, prompt-injected), THIS file alone
should still catch the most dangerous messages.

Design rules:

  1. ZERO-FALSE-NEGATIVE BIAS.
     We accept false positives ("someone said 'I'm dying of laughter'")
     in exchange for catching real crises. If a rule occasionally
     misfires, the user sees helpline numbers — that's recoverable.
     If a real crisis is missed, that's not.

  2. NO LLM HERE.
     Every detection in this file is a regex or substring match. That
     means it cannot be prompt-injected, cannot hallucinate, cannot
     fail because the model is overloaded. It runs in microseconds.

  3. WORDS PATTERNS, NOT NLP.
     We're not trying to be clever. We list the actual phrases people
     use in crisis. The LLM layer (crisis_llm.py) handles cleverness.

  4. EXPLICIT TEST COVERAGE.
     Every pattern added here MUST come with a test case in
     tests/test_crisis_rules.py. The eval harness in Task #7 will
     enforce this.

  5. INPUT NORMALIZATION.
     We lowercase + strip punctuation before matching, so "I.WANT.TO.DIE"
     and "I want to die!" both hit. Don't add fancy normalization beyond
     this — keep the function dead-simple to reason about.

Categories of patterns:
  - SUICIDE_PATTERNS: direct or near-direct ideation of self-harm/suicide
  - SELF_HARM_PATTERNS: cutting, hurting oneself
  - VIOLENCE_PATTERNS: danger to others (out of scope for support; route)
  - ABUSE_PATTERNS: being harmed by someone else (route to specific lines)

For now we collapse all categories into a single `is_crisis` signal —
the response is the same: show helplines, don't run the LLM. In a future
iteration we could route to different helpline numbers based on category
(e.g. iCall vs. women's helpline vs. childline).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

log = logging.getLogger("mannsaathi.agents.crisis_rules")


# ---- Pattern definitions ----------------------------------------------------
#
# Each pattern is a compiled regex with word-boundary or phrase matching.
# Patterns are explicit phrases, not stems, so we can review/audit them.

# Phrases that, when nearby, indicate the user is talking ABOUT the topic
# rather than expressing personal ideation. If any of these appear in the
# same message as a suicide/self-harm trigger phrase, we DON'T flag.
# Conservative list — we'd rather flag a borderline case than miss a real one.
_DISCUSSION_CONTEXT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\barticle (about|on)\b"),
    re.compile(r"\bbook (about|on)\b"),
    re.compile(r"\bdocumentary\b"),
    re.compile(r"\bnews (about|story)\b"),
    # "help my brother", "to support a friend", "supporting my sister", etc.
    re.compile(r"\b(help(ing)?|support(ing)?) (my|a|an|the) (brother|sister|friend|child|partner|colleague|son|daughter|mom|dad|mother|father)\b"),
    re.compile(r"\bhow can i help\b"),
    re.compile(r"\b(suicide|self.?harm) prevention\b"),
    re.compile(r"\bawareness (campaign|month|day)\b"),
]


# Suicide / suicidal ideation
_SUICIDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bkill (myself|my self)\b"),
    re.compile(r"\b(commit )?suicide\b"),
    re.compile(r"\bend (my|it|the) life\b"),
    re.compile(r"\bend it all\b"),
    re.compile(r"\bend (my|it) (all|now)\b"),
    re.compile(r"\bi want to die\b"),
    re.compile(r"\bi wanna die\b"),
    re.compile(r"\bwant to (just )?die\b"),
    # "dont want to live" with optional "anymore"
    re.compile(r"\bdont want to (live|be here|exist)( anymore)?\b"),
    re.compile(r"\bdo not want to (live|be here|exist)( anymore)?\b"),
    re.compile(r"\bno reason to (live|go on|continue)\b"),
    re.compile(r"\bbetter off (dead|without me|gone)\b"),
    re.compile(r"\b(everyone|world) would be better (off )?without me\b"),
    re.compile(r"\bi cant go on\b"),
    re.compile(r"\bi can not go on\b"),
    re.compile(r"\b(taking|going to take) my (own )?life\b"),
    re.compile(r"\bjump off (a |the )?(bridge|building|roof)\b"),
    re.compile(r"\bhang myself\b"),
    re.compile(r"\boverdose\b"),
    re.compile(r"\bsleeping pills\b.*\b(many|all|whole)\b"),
    re.compile(r"\b(many|all|whole)\b.*\bsleeping pills\b"),
    re.compile(r"\bthinking (of|about) (suicide|killing myself|ending it)\b"),
    re.compile(r"\bsuicidal\b"),
]

# Self-harm (cutting, hurting oneself, but not necessarily ending life)
_SELF_HARM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcut(ting)? (myself|my arm|my wrist|my leg)\b"),
    # "harm myself" / "harming myself" / "self-harming"
    re.compile(r"\bharm(ing)? (myself|my self)\b"),
    re.compile(r"\bhurt(ing)? (myself|my self)\b"),
    re.compile(r"\bself.?harm(ing)?\b"),
    re.compile(r"\bburn(ing)? (myself|my self)\b"),
]

# Imminent danger to others — out of scope for an AI companion; route immediately.
_VIOLENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(kill|hurt|harm) (him|her|them|someone|her self|him self)\b"),
    re.compile(r"\bgoing to (kill|shoot|stab|hurt)\b"),
]

# Being abused / in immediate danger from someone else.
# Includes explicit pronouns + common relationship nouns ("husband", "partner").
_ABUSE_RELATIONSHIPS = (
    "he|she|they|husband|wife|partner|boyfriend|girlfriend|"
    "father|mother|dad|mom|brother|sister|uncle|aunt|stepfather|stepmother"
)
_ABUSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        rf"\b(my )?({_ABUSE_RELATIONSHIPS}) (hit|hits|beat|beats|hurt|hurts|abuse|abuses) me\b"
    ),
    re.compile(r"\bbeing (abused|raped|assaulted)\b"),
    re.compile(r"\bsomeone is hurting me\b"),
]

# Combined list with category labels for diagnostics
_ALL_PATTERNS: list[tuple[str, re.Pattern[str]]] = (
    [("suicide", p) for p in _SUICIDE_PATTERNS]
    + [("self_harm", p) for p in _SELF_HARM_PATTERNS]
    + [("violence", p) for p in _VIOLENCE_PATTERNS]
    + [("abuse", p) for p in _ABUSE_PATTERNS]
)


# ---- Public API -------------------------------------------------------------

@dataclass
class RulesResult:
    """Outcome of running the rules layer on a single message."""

    is_crisis: bool
    matched_categories: List[str]  # e.g. ["suicide"] or ["self_harm", "suicide"]
    matched_patterns: List[str]    # raw pattern strings, for logging/debug


# Normalization (order matters):
#   1. Strip apostrophes / fancy quotes WITHOUT inserting a space, so
#      "don't" -> "dont" (NOT "don t"). This lets patterns written as
#      "dont want to live" match the contracted form too.
#   2. Replace remaining punctuation (commas, dashes, em-dashes, periods)
#      with spaces so "I.WANT.TO.DIE" -> "i want to die".
#   3. Collapse multiple whitespace into one space.
_NORMALIZE_QUOTE_RE = re.compile(r"['‘’“”`]")
_NORMALIZE_PUNCT_RE = re.compile(r"[^\w\s]")
_NORMALIZE_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _NORMALIZE_QUOTE_RE.sub("", text)   # don't -> dont
    text = _NORMALIZE_PUNCT_RE.sub(" ", text)  # I.want.to.die -> I want to die
    text = _NORMALIZE_WS_RE.sub(" ", text)
    return text.strip()


def _looks_like_discussion(normalized: str) -> bool:
    """True if the message frames the topic as discussion-about, not personal."""
    return any(p.search(normalized) for p in _DISCUSSION_CONTEXT_PATTERNS)


def check_rules(text: str) -> RulesResult:
    """
    Run all deterministic crisis patterns against the input text.

    Returns a RulesResult with is_crisis=True if ANY pattern matches AND
    the message doesn't look like abstract discussion of the topic.

    The discussion-context filter is conservative — it only suppresses
    the rules layer's flag. The LLM classifier (which has full context)
    runs independently and can still flag a real crisis that was
    misclassified as discussion. That's the value of having two layers.
    """
    normalized = _normalize(text)

    categories: list[str] = []
    matched: list[str] = []
    for category, pattern in _ALL_PATTERNS:
        if pattern.search(normalized):
            if category not in categories:
                categories.append(category)
            matched.append(pattern.pattern)

    # If a pattern matched but the message reads like discussion-about
    # (article, documentary, helping someone else, prevention awareness),
    # suppress the rules-layer flag. LLM layer still runs independently.
    if matched and _looks_like_discussion(normalized):
        log.info(
            "rules layer: pattern matched but context looks like discussion — suppressed "
            "| patterns=%s",
            matched,
        )
        return RulesResult(
            is_crisis=False, matched_categories=[], matched_patterns=[]
        )

    is_crisis = bool(matched)
    if is_crisis:
        log.info(
            "rules layer: CRISIS detected | categories=%s | patterns=%s",
            categories,
            matched,
        )
    return RulesResult(
        is_crisis=is_crisis,
        matched_categories=categories,
        matched_patterns=matched,
    )
