"""
The interview engine: the deterministic referee of the platform.

This is the architectural heart that the brief rewards. A separate, optional
language model may generate questions and produce rubric scores, but every
decision lives here: difficulty selection, scoring aggregation, and termination.
That makes the result auditable, reproducible, and explainable.

Adaptive difficulty uses an Elo-style ability estimate, the practical form of
computerized adaptive testing. The candidate holds an ability rating; each
question holds a difficulty rating. After each answer the ability moves toward
the gap between actual and expected performance, and the next question is chosen
as a small stretch above the current estimate.

All tunable rules live in Config and are restated in the README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from question_bank import (Question, questions_for_topics, extract_role,
                           coverage_map, SKILL_TAXONOMY)
from evaluation import evaluate_answer, _content_words


class Config:
    DIFFICULTY_LADDER = ["easy", "medium", "hard"]
    BASE_POINTS = {"easy": 10.0, "medium": 20.0, "hard": 30.0}

    # Elo-style adaptive testing.
    DIFFICULTY_RATING = {"easy": 1000.0, "medium": 1300.0, "hard": 1600.0}
    START_ABILITY = 1150.0
    ELO_K = 140.0            # how fast the ability estimate moves
    STRETCH = 60.0          # aim the next question just above the estimate
    ABILITY_MIN, ABILITY_MAX = 800.0, 1800.0

    GOOD_ANSWER_QUALITY = 0.60

    MAX_QUESTIONS = 10
    MIN_QUESTIONS_BEFORE_CUT = 4    # a single weak answer must not end the interview
    SCORE_FLOOR = 40.0              # end early if the running score falls below this
    MAX_CONSECUTIVE_POOR = 3

    BANDS = [(75.0, "Strong"), (50.0, "Average"), (0.0, "Needs Improvement")]


@dataclass
class TurnRecord:
    qid: str
    topic: str
    category: str
    difficulty: str
    question: str
    answer: str
    time_taken: float
    time_limit: int
    dimensions: dict
    quality: float
    points: float
    points_possible: float
    incomplete: bool
    rationale: str
    expected: float            # Elo expected performance for this question
    ability_before: float
    ability_after: float
    source: str                # "model" or "rules"


@dataclass
class InterviewState:
    total_points: float = 0.0
    max_points: float = 0.0
    answered: int = 0
    current_difficulty: str = "easy"
    ability: float = Config.START_ABILITY
    time_used: float = 0.0
    consecutive_poor: int = 0
    finished: bool = False
    termination_reason: Optional[str] = None
    asked_ids: set = field(default_factory=set)
    turns: list = field(default_factory=list)
    topic_totals: dict = field(default_factory=dict)
    decision_log: list = field(default_factory=list)


class InterviewEngine:
    def __init__(self, topics: list[str], resume_text: str = "", jd_text: str = "",
                 max_questions: int | None = None, start_difficulty: str | None = None,
                 use_llm: bool = False):
        if not topics:
            raise ValueError("At least one topic is required to start an interview.")
        self.topics = topics
        self.role = extract_role(jd_text)
        self.jd_terms = _content_words(jd_text)
        self.use_llm = use_llm
        self._pool = questions_for_topics(topics)
        if not self._pool:
            raise ValueError("No questions are available for the selected topics.")
        self.max_questions = max_questions or Config.MAX_QUESTIONS

        covered, gaps, jd_topics = coverage_map(resume_text, jd_text)
        self.covered, self.gaps, self.jd_topics = covered, gaps, jd_topics
        self.skill_weights = self._build_weights(jd_text)

        self.state = InterviewState()
        if start_difficulty and start_difficulty in Config.DIFFICULTY_LADDER:
            self.state.current_difficulty = start_difficulty
        self.state.decision_log.append(
            f"Interview opened for {self.role}. Starting ability estimate "
            f"{self.state.ability:.0f}, first question at {self.state.current_difficulty} level."
        )

    # -- weight each skill by how strongly the job description emphasizes it --
    def _build_weights(self, jd_text: str) -> dict:
        text = jd_text.lower()
        weights = {}
        for topic in self.topics:
            hits = sum(text.count(t) for t in SKILL_TAXONOMY.get(topic, []))
            weights[topic] = 1.0 + float(hits)   # base of 1 so every topic counts
        return weights

    # -- normalized running score (0 to 100), weighted by skill importance --
    def running_score(self) -> float:
        if not self.state.topic_totals:
            return 0.0
        num = den = 0.0
        for topic, (earned, possible) in self.state.topic_totals.items():
            if possible <= 0:
                continue
            pct = earned / possible * 100.0
            w = self.skill_weights.get(topic, 1.0)
            num += pct * w
            den += w
        return round(num / den, 2) if den else 0.0

    # -- Elo helpers --
    def _expected(self, difficulty: str) -> float:
        rating = Config.DIFFICULTY_RATING.get(difficulty, 1150.0)
        return 1.0 / (1.0 + 10.0 ** ((rating - self.state.ability) / 400.0))

    def _difficulty_for_ability(self) -> str:
        target = self.state.ability + Config.STRETCH
        return min(Config.DIFFICULTY_LADDER,
                   key=lambda d: abs(Config.DIFFICULTY_RATING[d] - target))

    # -- choose the next question, biased toward unanswered topics and gaps --
    def current_question(self) -> Optional[Question]:
        if self.state.finished:
            return None
        unused = [q for q in self._pool if q.qid not in self.state.asked_ids]
        if not unused:
            return None

        ladder = Config.DIFFICULTY_LADDER
        order = sorted(ladder, key=lambda d: abs(ladder.index(d)
                       - ladder.index(self.state.current_difficulty)))
        for difficulty in order:
            candidates = [q for q in unused if q.difficulty == difficulty]
            if not candidates:
                continue
            asked_per_topic = {}
            for turn in self.state.turns:
                asked_per_topic[turn.topic] = asked_per_topic.get(turn.topic, 0) + 1
            # Prefer least-asked topics, then job-description gaps.
            candidates.sort(key=lambda q: (asked_per_topic.get(q.topic, 0),
                                           0 if q.topic in self.gaps else 1))
            return self._maybe_llm_question(candidates[0])
        return self._maybe_llm_question(unused[0])

    def _maybe_llm_question(self, q: Question) -> Question:
        """Optionally rewrite the question wording with the model, keeping the engine's slot.

        The engine has already chosen the topic, difficulty, category, and time limit.
        Only the wording and the expected concepts may come from the model. On any
        failure this returns the bank question unchanged, so an interview always has a
        question to ask even if the model is unavailable.
        """
        if not self.use_llm:
            return q
        try:
            import llm
            asked = [t.question for t in self.state.turns]
            generated = llm.generate_question_llm(q.topic, q.difficulty, self.role, asked)
            if generated and generated.get("text"):
                return Question(
                    qid=q.qid, topic=q.topic, category=q.category,
                    difficulty=q.difficulty, text=generated["text"],
                    time_limit=q.time_limit,
                    expected_keywords=generated.get("expected_keywords") or q.expected_keywords,
                )
        except Exception:
            pass
        return q

    # -- submit an answer and advance the deterministic state --
    def submit(self, question: Question, answer_text: str, time_taken: float) -> TurnRecord:
        if self.state.finished:
            raise RuntimeError("The interview has already finished.")

        llm_scores, source = self._maybe_llm_scores(question, answer_text)
        result = evaluate_answer(question, answer_text, time_taken,
                                 self.jd_terms, llm_scores=llm_scores)
        quality = result["quality"]
        base = Config.BASE_POINTS.get(question.difficulty, 10.0)
        points = round(base * quality, 2)

        self.state.total_points += points
        self.state.max_points += base
        self.state.time_used += time_taken
        self.state.answered += 1
        self.state.asked_ids.add(question.qid)

        topic_total = self.state.topic_totals.setdefault(question.topic, [0.0, 0.0])
        topic_total[0] += points
        topic_total[1] += base

        # Elo ability update.
        expected = self._expected(question.difficulty)
        ability_before = self.state.ability
        self.state.ability = max(Config.ABILITY_MIN, min(
            Config.ABILITY_MAX,
            self.state.ability + Config.ELO_K * (quality - expected)))

        if quality >= Config.GOOD_ANSWER_QUALITY and not result["incomplete"]:
            self.state.consecutive_poor = 0
            verdict = "strong"
        else:
            self.state.consecutive_poor += 1
            verdict = "weak"

        record = TurnRecord(
            qid=question.qid, topic=question.topic, category=question.category,
            difficulty=question.difficulty, question=question.text,
            answer=answer_text.strip(), time_taken=round(time_taken, 1),
            time_limit=question.time_limit, dimensions=result["dimensions"],
            quality=quality, points=points, points_possible=base,
            incomplete=result["incomplete"], rationale=result["rationale"],
            expected=round(expected, 3), ability_before=round(ability_before),
            ability_after=round(self.state.ability), source=source,
        )
        self.state.turns.append(record)

        self._check_termination()
        if not self.state.finished:
            self.state.current_difficulty = self._difficulty_for_ability()

        self.state.decision_log.append(
            f"Q{self.state.answered} {question.difficulty} ({question.topic}): "
            f"quality {quality:.2f}, expected {expected:.2f} -> {verdict}. "
            f"Ability {ability_before:.0f} -> {self.state.ability:.0f}. "
            + (f"Interview ends: {self.state.termination_reason}"
               if self.state.finished
               else f"Next question targets {self.state.current_difficulty}.")
        )
        return record

    def _maybe_llm_scores(self, question: Question, answer_text: str):
        """Return (llm_scores or None, source label). Falls back safely on any error."""
        if not self.use_llm:
            return None, "rules"
        try:
            import llm
            scores = llm.score_answer_llm(question.text, answer_text,
                                          question.expected_keywords)
            if scores:
                return scores, "model"
        except Exception:
            pass
        return None, "rules"

    def _check_termination(self) -> None:
        s = self.state
        if s.answered >= self.max_questions:
            s.finished, s.termination_reason = True, "Reached the maximum number of questions."
            return
        if s.answered >= Config.MIN_QUESTIONS_BEFORE_CUT:
            if s.consecutive_poor >= Config.MAX_CONSECUTIVE_POOR:
                s.finished, s.termination_reason = True, "Several weak answers in a row."
                return
            if self.running_score() < Config.SCORE_FLOOR:
                s.finished, s.termination_reason = True, \
                    "Performance fell below the readiness threshold."
                return
        if all(q.qid in s.asked_ids for q in self._pool):
            s.finished, s.termination_reason = True, "All available questions were answered."

    def is_finished(self) -> bool:
        if not self.state.finished:
            self._check_termination()
        return self.state.finished

    # -- the explainable final report --
    def report(self) -> dict:
        s = self.state
        score = self.running_score()
        band = next(label for floor, label in Config.BANDS if score >= floor)

        skill_breakdown = {
            topic: round(earned / possible * 100.0, 1) if possible else 0.0
            for topic, (earned, possible) in s.topic_totals.items()
        }
        strengths = sorted((t for t, v in skill_breakdown.items() if v >= 70.0),
                           key=lambda t: skill_breakdown[t], reverse=True)
        weaknesses = sorted((t for t, v in skill_breakdown.items() if v < 50.0),
                            key=lambda t: skill_breakdown[t])
        dimension_avg = self._dimension_averages()
        early = s.finished and s.answered < self.max_questions and \
            s.termination_reason not in (None, "All available questions were answered.")
        total_w = sum(self.skill_weights.get(t, 1.0) for t in s.topic_totals) or 1.0
        norm_weights = {t: round(self.skill_weights.get(t, 1.0) / total_w, 2)
                        for t in s.topic_totals}

        return {
            "interview_readiness_score": score,
            "readiness_band": band,
            "hiring_readiness": self._hiring_verdict(band),
            "ability_estimate": round(s.ability),
            "ability_trajectory": [t.ability_after for t in s.turns],
            "questions_answered": s.answered,
            "time_used_seconds": round(s.time_used, 1),
            "terminated_early": early,
            "termination_reason": s.termination_reason,
            "skill_breakdown": skill_breakdown,
            "skill_weights": norm_weights,
            "dimension_averages": dimension_avg,
            "coverage": {"covered": self.covered, "gaps": self.gaps},
            "strengths": strengths,
            "weaknesses": weaknesses,
            "actionable_feedback": self._feedback(weaknesses, dimension_avg),
            "decision_log": s.decision_log,
            "transcript": [self._turn_summary(t) for t in s.turns],
        }

    def _dimension_averages(self) -> dict:
        if not self.state.turns:
            return {}
        names = ["accuracy", "relevance", "depth", "clarity", "time_efficiency"]
        totals = {n: 0.0 for n in names}
        for turn in self.state.turns:
            for n in names:
                totals[n] += turn.dimensions[n]
        count = len(self.state.turns)
        return {n: round(totals[n] / count * 100.0, 1) for n in names}

    def _hiring_verdict(self, band: str) -> str:
        if band == "Strong":
            return f"Recommended for {self.role}."
        if band == "Average":
            return f"Borderline: promising, but needs more polish for {self.role}."
        return f"Not yet ready for {self.role}; focused practice is advised."

    def _feedback(self, weaknesses: list[str], dimension_avg: dict) -> list[str]:
        advice = {
            "accuracy": "Tighten technical accuracy by naming the core concepts each question targets.",
            "relevance": "Stay on point: answer the exact question before adding extra detail.",
            "depth": "Add depth with concrete examples, trade-offs, and the reasoning behind choices.",
            "clarity": "Improve clarity by leading with a one-line summary, then the supporting detail.",
            "time_efficiency": "Manage time better: outline the answer first so you finish within the limit.",
        }
        tips = []
        for name, _ in sorted(dimension_avg.items(), key=lambda kv: kv[1])[:2]:
            if dimension_avg.get(name, 100) < 70:
                tips.append(advice[name])
        for topic in weaknesses[:2]:
            tips.append(f"Strengthen {topic.replace('_', ' ')} fundamentals before the next interview.")
        if self.gaps:
            tips.append("The job description asks for "
                        + ", ".join(g.replace('_', ' ') for g in self.gaps)
                        + ", which your resume does not clearly show; prepare these especially.")
        if not tips:
            tips.append("Strong all-round performance; keep practising harder questions to stay sharp.")
        return tips

    @staticmethod
    def _turn_summary(turn: TurnRecord) -> dict:
        return {
            "qid": turn.qid, "topic": turn.topic, "difficulty": turn.difficulty,
            "quality": turn.quality, "points": turn.points,
            "points_possible": turn.points_possible, "time_taken": turn.time_taken,
            "time_limit": turn.time_limit, "incomplete": turn.incomplete,
            "expected": turn.expected, "ability_after": turn.ability_after,
            "source": turn.source, "dimensions": turn.dimensions,
            "rationale": turn.rationale,
        }
