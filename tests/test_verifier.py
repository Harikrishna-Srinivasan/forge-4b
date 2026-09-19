from forge.verifier import verify_python, check_constraints, contradiction_scan, judge_pick

def test_verify_python_ok():
    assert verify_python("def add(a,b):\n return a+b", "\nassert add(2,3)==5\n")["ok"]

def test_verify_python_fail():
    assert not verify_python("def add(a,b):\n return a-b", "\nassert add(2,3)==5\n")["ok"]

def test_constraints():
    assert check_constraints("hello commit", ["commit"], ["push"])["ok"]
    assert not check_constraints("hello push", ["commit"], ["push"])["ok"]

def test_contradiction():
    r = contradiction_scan("All birds fly. However penguins never fly except sometimes.")
    assert "absolute-then-hedge" in r["flags"]

def test_judge_prefers_ok():
    c = [{"answer": "long wrong " * 50}, {"answer": "short right"}]
    v = [{"ok": False}, {"ok": True}]
    assert judge_pick(c, v) == 1
