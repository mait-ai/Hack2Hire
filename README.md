# AI-Powered Mock Interview Platform

A mock interview platform that reads a candidate resume and a job description, runs
a timed and adaptive interview, and produces an explainable Interview Readiness
Score. Built for the Hack2Hire hackathon.

The design follows the principle that wins this brief: a deterministic referee
owns every decision, while an optional language model is a constrained component
that only proposes questions or returns rubric scores. This keeps the result
auditable, reproducible, and explainable.

## Demo Video (required)

> Replace this line with a link to your screen recording of the live, working
> platform. The submission rules require this video to appear in the README.
> Record one strong-candidate run and one weak-candidate run so the adaptive
> behaviour and the early termination are both visible.

## What it does

- Accepts a resume and a job description by file upload (PDF or text) or by pasting,
  and performs a true gap analysis: which required skills the resume covers, and
  which are gaps to probe.
- Asks technical, conceptual, behavioral, and scenario questions selected for the role.
- Adapts difficulty with an Elo-style ability estimate, the practical form of
  computerized adaptive testing.
- Enforces a visible time limit per question and penalizes over-time answers.
- Ends the interview early when performance falls below a threshold, after a
  minimum number of questions.
- Scores each answer on accuracy, clarity, depth, relevance, and time efficiency.
- Shows a live transparency dashboard: the current ability estimate, the chosen
  difficulty, the latest per-dimension scores, and a running decision log.
- Produces a final readiness score, a per-skill breakdown, strengths and
  weaknesses, actionable feedback, and a role-specific hiring verdict.

## Architecture

- `engine.py` is the deterministic referee. It owns the Elo ability update,
  difficulty selection, scoring aggregation, termination rules, the decision log,
  and the final report.
- `evaluation.py` scores one answer across the five dimensions. It uses
  transparent heuristics by default, and can accept model rubric scores instead;
  either way the engine performs the aggregation.
- `question_bank.py` holds the tagged question bank, the skill taxonomy, the
  resume and job-description analysis, and the coverage map.
- `llm.py` is the optional OpenAI layer. The app works fully without it.
- `app.py` is the Streamlit interface, including the live transparency dashboard.
- `test_core.py` verifies the rules and the edge cases.

## Scoring and decision rules (explainable)

These exact rules live in `Config` in `engine.py` and in `DIMENSION_WEIGHTS` in
`evaluation.py`.

- Difficulty ratings: easy 1000, medium 1300, hard 1600. Base points: easy 10,
  medium 20, hard 30.
- Ability estimate: starts at 1150, bounded between 800 and 1800. Expected
  performance equals one divided by one plus ten raised to the power of the
  difficulty rating minus the ability, divided by four hundred. After each answer
  the ability moves by a factor of 140 times the gap between the actual quality
  and the expected value. The next question targets the difficulty nearest to the
  ability plus a stretch of 60.
- Dimension weights: accuracy 0.35, relevance 0.20, depth 0.20, clarity 0.10,
  time efficiency 0.15. These produce a quality value from 0 to 1.
- Points for an answer equal the base points for its difficulty times the quality.
- Final score: a skill-importance-weighted average of the per-skill percentages,
  where each skill is weighted by how strongly the job description emphasizes it.
- Readiness bands: 75 and above is Strong, 50 and above is Average, below 50 is
  Needs Improvement.
- Time: an answer within the limit scores on how well the time was used. An answer
  over the limit scores zero on time efficiency and is flagged as incomplete, while
  its content still earns partial credit.
- Termination, after at least four questions: three weak answers in a row, or the
  running score below 40. Also when the maximum question count is reached, or the
  question pool is exhausted. A single weak answer can never end the interview.

## Edge cases handled

Verified by `test_core.py`: blank answers, "I do not know" answers, over-time
answers with partial credit, attempts to manipulate the score through instructions
inside the answer, very long and very short answers, a resume with no relevant
skills, a strong candidate who does not exhaust the questions, and a minimum
question count before any early termination.

## Optional OpenAI layer

The platform is fully functional without any key. To enable model-generated
questions and model-based rubric scoring:

- Provide a key as an environment variable named `OPENAI_API_KEY`, or, on
  Streamlit Community Cloud, under Settings, then Secrets, as
  `OPENAI_API_KEY = "sk-..."`.
- Never hard-code a key in the source, and never commit a key to a public
  repository.
- The model is constrained: it returns only rubric scores or a question, as JSON,
  at a low temperature, and the prompt instructs it to ignore any instruction
  contained in a candidate answer. The deterministic engine still owns the final
  score. If the key is missing or a call fails, the platform falls back to the
  built-in scoring automatically.

## A note on camera and facial-expression scoring

This was considered and deliberately not included. Reliable webcam emotion
recognition is difficult to run on free hosting, it would add significant failure
risk, and scoring candidates on facial expressions is scientifically contested and
restricted in real hiring in several jurisdictions. The platform instead invests in
the adaptive engine, the transparency dashboard, and the gap analysis, which is
where the brief states the marks are awarded.

## Tech stack

- Python 3 and Streamlit.
- The standard library only for the engine, the evaluator, and the question bank.
- OpenAI is optional and used only if a key is configured.

## Setup and run

1. Install dependencies: `pip install -r requirements.txt`
2. Start the platform: `streamlit run app.py`
3. Upload a resume and a job description as PDF or text, or paste them, or use the
   sample button, then start.

To run the rule and edge-case checks without the interface: `python test_core.py`

## Optional: deploy for an edge

Streamlit Community Cloud deploys free from a public GitHub repository. Connect the
repository and point it at `app.py`. Add the OpenAI key under Secrets only if you
want the optional model scoring.

## Project structure

- `app.py`, `engine.py`, `evaluation.py`, `question_bank.py`, `llm.py`
- `test_core.py`, `requirements.txt`
- `sample_resume.txt`, `sample_jd.txt`
