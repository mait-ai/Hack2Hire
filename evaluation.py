"""
Heuristic answer evaluation.

Each answer is scored on five dimensions, every one normalized to the range
0.0 to 1.0, and then combined into a single quality value using fixed weights.

    accuracy        coverage of the concepts the question expects
    relevance       overlap with the question and the job description
    depth           elaboration, measured by length and concept variety
    clarity         sentence structure and readability
    time_efficiency how well the time limit was used

This logic is fully deterministic and transparent, which is what makes the
final score explainable. Replace evaluate_answer with a model-based version
later if desired; the rest of the system depends only on its return shape.
"""

from __future__ import annotations

import re

from question_bank import Question


# Weights sum to 1.0. Documented in the README so scoring stays explainable.
DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "relevance": 0.20,
    "depth": 0.20,
    "clarity": 0.10,
    "time_efficiency": 0.15,
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "it", "this",
    "that", "these", "those", "i", "you", "he", "she", "they", "we", "my",
    "your", "what", "how", "why", "when", "which", "do", "does", "would",
    "can", "could", "should", "about", "into", "from", "its", "their",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'+-]*", text.lower())


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOPWORDS and len(w) > 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _score_accuracy(question: Question, answer_text: str) -> float:
    """Fraction of the expected concepts that appear in the answer."""
    if not question.expected_keywords:
        return -1.0  # signals "no expected concepts"; caller substitutes relevance
    text = answer_text.lower()
    matched = sum(1 for kw in question.expected_keywords if kw.lower() in text)
    return _clamp(matched / len(question.expected_keywords))


def _score_relevance(question: Question, answer_text: str, jd_terms: set[str]) -> float:
    """Coverage of the question's terms, plus a contribution for job-description terms."""
    answer_terms = _content_words(answer_text)
    if not answer_terms:
        return 0.0
    q_terms = _content_words(question.text)
    q_cov = len(answer_terms & q_terms) / max(1, len(q_terms))
    jd_overlap = len(answer_terms & jd_terms)
    jd_component = _clamp(jd_overlap / 3.0)
    return _clamp(0.6 * q_cov + 0.4 * jd_component)


def _score_depth(question: Question, answer_text: str) -> float:
    """Elaboration: a blend of answer length and the variety of concepts touched."""
    word_count = len(_words(answer_text))
    length_component = _clamp(word_count / 70.0)

    if question.expected_keywords:
        text = answer_text.lower()
        distinct = sum(1 for kw in question.expected_keywords if kw.lower() in text)
        concept_component = _clamp(distinct / 3.0)
    else:
        # For behavioral answers, reward structure through distinct content words.
        concept_component = _clamp(len(_content_words(answer_text)) / 25.0)

    return _clamp(0.5 * length_component + 0.5 * concept_component)


def _score_clarity(answer_text: str) -> float:
    """Reward well-formed sentences of a readable length."""
    sentences = _sentences(answer_text)
    if not sentences:
        return 0.0
    total_words = len(_words(answer_text))
    avg_len = total_words / len(sentences)

    # Ideal average sentence length is roughly eight to twenty-two words.
    if avg_len < 4:
        length_quality = 0.4
    elif avg_len <= 22:
        length_quality = 1.0
    elif avg_len <= 32:
        length_quality = 0.7
    else:
        length_quality = 0.4

    structure_bonus = 0.1 if len(sentences) >= 2 else 0.0
    return _clamp(length_quality + structure_bonus)


def _score_time(time_taken: float, time_limit: float) -> tuple[float, bool]:
    """Return (time_efficiency, incomplete). Over-time answers score zero and are flagged."""
    if time_limit <= 0:
        return 0.5, False
    if time_taken > time_limit:
        return 0.0, True
    ratio = time_taken / time_limit
    if ratio < 0.2:
        return 0.6, False      # very fast, likely shallow
    if ratio <= 0.85:
        return 1.0, False      # good use of the available time
    return 0.8, False          # close to the limit


def evaluate_answer(question: Question, answer_text: str,
                    time_taken: float, jd_terms: set[str] | None = None,
                    llm_scores: dict | None = None) -> dict:
    """Score one answer into the five dimensions, a combined quality, and a rationale.

    If llm_scores is supplied (accuracy, clarity, depth, relevance, each in 0..1),
    those content scores are used; otherwise the deterministic heuristics are used.
    Time efficiency and the final aggregation are always computed here, so the
    deterministic referee owns the score regardless of where the content scores came from.
    """
    jd_terms = jd_terms or set()
    answer_text = (answer_text or "").strip()
    time_efficiency, incomplete = _score_time(time_taken, question.time_limit)

    if llm_scores:
        accuracy = _clamp(float(llm_scores.get("accuracy", 0.0)))
        clarity = _clamp(float(llm_scores.get("clarity", 0.0)))
        depth = _clamp(float(llm_scores.get("depth", 0.0)))
        relevance = _clamp(float(llm_scores.get("relevance", 0.0)))
    else:
        relevance = _score_relevance(question, answer_text, jd_terms)
        accuracy = _score_accuracy(question, answer_text)
        if accuracy < 0.0:
            accuracy = relevance  # behavioral questions: relevance stands in for accuracy
        depth = _score_depth(question, answer_text)
        clarity = _score_clarity(answer_text)

    if not answer_text:
        accuracy = relevance = depth = clarity = 0.0
        time_efficiency = 0.0
        incomplete = True

    dimensions = {
        "accuracy": round(accuracy, 3),
        "relevance": round(relevance, 3),
        "depth": round(depth, 3),
        "clarity": round(clarity, 3),
        "time_efficiency": round(time_efficiency, 3),
    }
    quality = sum(dimensions[name] * weight for name, weight in DIMENSION_WEIGHTS.items())
    quality = round(_clamp(quality), 3)

    return {
        "dimensions": dimensions,
        "quality": quality,
        "incomplete": incomplete,
        "rationale": _build_rationale(question, dimensions, incomplete),
    }


def _build_rationale(question: Question, dimensions: dict, incomplete: bool) -> str:
    parts = []
    if question.expected_keywords:
        covered = round(dimensions["accuracy"] * len(question.expected_keywords))
        parts.append(f"covered about {covered} of {len(question.expected_keywords)} key concepts")
    else:
        parts.append(f"relevance {int(dimensions['relevance'] * 100)} percent")
    parts.append("clear structure" if dimensions["clarity"] >= 0.7 else "structure could improve")
    if incomplete:
        parts.append("ran over the time limit")
    elif dimensions["time_efficiency"] >= 0.9:
        parts.append("used time well")
    return "; ".join(parts).capitalize() + "."
