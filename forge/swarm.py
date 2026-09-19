"""Teacher/critic/data-engineering swarm: MiMo=teacher, Muse Spark=researcher/curator,
Big Pickle=adversarial critic. Student is qwen3:4b (4B, no fine-tune locally).
Free LLMs produce+judge DATA, never 'train the model' directly."""
from __future__ import annotations
import json
from .student import Student
from .verifier import verify_python, check_constraints

MIMO_TEACHER_SYS = ("You are the expert teacher for Forge-4B. Solve as accurately as possible. "
 "For coding/terminal: inspect before assuming, use tools when available, verify, recover from failures. "
 "Return: 1.actions 2.observations 3.hypotheses 4.final solution 5.verification 6.likely failure modes. "
 "Do not invent successful execution./no_think")

MUSE_RESEARCHER_SYS = ("You are the research and dataset engineer for Forge-4B (Muse Spark). "
 "Given task/candidate trajectory: identify capability, mistakes, useful patterns, harder variants, "
 "remove low-value examples, convert to training records. Prefer verified evidence over plausible text./no_think")

BIGPICKLE_CRITIC_SYS = ("You are Forge-4B's adversarial verifier (Big Pickle). Never assume answer correct. "
 "Find factual errors, logical gaps, violated instructions, hidden assumptions, counterexamples, "
 "bad tool use, unnecessary actions. Give concrete falsifying test. Return: PASS/FAIL, failure_reason, "
 "earliest_error, verification, corrected_solution, training_lesson./no_think")

class Swarm:
    def __init__(self, model_teacher="kamekichi128/qwen3-4b-instruct-2507:latest", model_student="kamekichi128/qwen3-4b-instruct-2507:latest"):
        self.teacher = Student(model_teacher)   # MiMo-role proxy (deepseek reasoning) w/ fallback
        self.student = Student(model_student)   # Forge-4B student
        self.critic = Student(model_student)    # Big Pickle-role proxy
        self.curator = Student(model_student)   # Muse-role proxy (this session is Muse Spark)
        self.teacher_model = model_teacher
        self.student_model = model_student

    def mimo_solve(self, prompt: str) -> dict:
        import os
        if os.environ.get("FORGE_ZEN", "1") == "1":
            try:
                from .zen_teachers import ask, DEFAULTS
                z = ask(prompt, DEFAULTS[0], MIMO_TEACHER_SYS)
                if z.get("ok"):
                    return {"agent": f"mimo-teacher:{z['model']}(zen-cloud)",
                            "answer": z["answer"],
                            "license": "unknown-do-not-redistribute"}
            except Exception:
                pass
        try: ans = self.teacher.generate(prompt, system=MIMO_TEACHER_SYS, temperature=0.7)
        except Exception:  # fallback to student if teacher model busy
            ans = self.student.generate(prompt, system=MIMO_TEACHER_SYS, temperature=0.7)
        return {"agent": f"mimo-teacher:{self.teacher_model}", "answer": ans}

    def student_attempt(self, prompt: str) -> dict:
        ans = self.student.generate(prompt, temperature=0.7)
        return {"agent": f"forge-4b-student:{self.student_model}", "answer": ans}

    def bigpickle_critique(self, problem: str, candidates: list[dict]) -> dict:
        joined = "\n\n".join(f"[{c['agent']}]:\n{c['answer'][:1500]}" for c in candidates)
        # deterministic pre-check: execute python blocks when present
        import re
        det = []
        for c in candidates:
            m = re.findall(r"```python(.*?)```", c["answer"] or "", re.S)
            if m: det.append(verify_python(m[0].strip(), ""))
        prompt = (f"Problem: {problem}\nCandidates:\n{joined}\nCompare A/B/C. Which is correct, why, "
                  "what evidence distinguishes them, construct a test. Return PASS/FAIL + lesson./no_think")
        import os
        if os.environ.get("FORGE_ZEN", "1") == "1":
            try:
                from .zen_teachers import ask, DEFAULTS
                cm = DEFAULTS[1] if len(DEFAULTS) > 1 else DEFAULTS[0]
                z = ask(prompt, cm, BIGPICKLE_CRITIC_SYS)
                if z.get("ok"):
                    uniq = len({(c.get("answer") or "")[:200] for c in candidates})
                    return {"agent": f"bigpickle-critic:{cm}(zen-cloud)", "verdict": z["answer"],
                            "disagreement": uniq > 1, "n_unique": uniq, "deterministic": det}
            except Exception:
                pass
        try: verdict = self.critic.generate(prompt, system=BIGPICKLE_CRITIC_SYS, temperature=0.4)
        except Exception as e: verdict = f"critic-failed: {e}"
        # disagreement signal: distinct answers -> interesting
        uniq = len({(c.get("answer") or "")[:200] for c in candidates})
        return {"agent": "bigpickle-critic", "verdict": verdict,
                "disagreement": uniq > 1, "n_unique": uniq,
                "deterministic": det}

    def muse_curate(self, task: dict, result: dict) -> dict:
        prompt = (f"Task: {json.dumps(task)[:1200]}\nResult: {json.dumps(result)[:2500]}\n"
                  "Identify capability, mistakes, useful pattern, harder variant, keep/drop, training record./no_think")
        try: rec = self.curator.generate(prompt, system=MUSE_RESEARCHER_SYS, temperature=0.5)
        except Exception as e: rec = f"curator-failed: {e}"
        return {"agent": "muse-researcher", "record": rec}

def make_failure_lesson(task: dict, bad_approach: str, observation: str,
                        correction: str, lesson: str) -> dict:
    return {"task": task.get("prompt", "")[:800], "BAD_APPROACH": bad_approach[:800],
            "OBSERVATION": observation[:800], "CORRECTION": correction[:800],
            "LESSON": lesson[:800], "type": "failure_lesson"}
