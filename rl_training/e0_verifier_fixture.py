# E0 code-verifier audit (§56): run a compact fixture of KNOWN programs through the code verifier, both
# SEQUENTIAL and POOLED, in this pod. Separates infra-failures from wrong answers. Diagnoses the §55-D2
# implausibly-low verified-harvest bug (pooled subprocess returns 0 in the shared-PID container).
# Usage: python -m rl_training.e0_verifier_fixture
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.code_passk import run_tests

# fixture: (code, item, expected_ok, label)
FIX = [
    ("def add(a,b):\n    return a+b\n",
     {"test": "assert add(2,3)==5\nassert add(-1,1)==0", "entry": "add", "mbpp": True}, True, "correct"),
    ("def add(a,b):\n    return a-b\n",
     {"test": "assert add(2,3)==5", "entry": "add", "mbpp": True}, False, "wrong-answer"),
    ("def boom(a):\n    raise ValueError('x')\n",
     {"test": "assert boom(1)==1", "entry": "boom", "mbpp": True}, False, "exception"),
    ("import time\ndef slow(a):\n    time.sleep(30)\n    return a\n",
     {"test": "assert slow(1)==1", "entry": "slow", "mbpp": True}, False, "timeout"),
    ("def f(x):\n    return sorted(x)\n",
     {"test": "assert candidate([3,1,2])==[1,2,3]", "entry": "f",
      "prompt": ""}, True, "humaneval-style-check"),
]

def seq_run():
    out = []
    for code, item, exp, lab in FIX:
        ok, err = run_tests(code, item, timeout=5, return_err=True)
        out.append((lab, ok, exp, ok == exp, err[:40]))
    return out

def pooled_run():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = [None] * len(FIX)
    try:
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(run_tests, c, it, 5, True): i for i, (c, it, e, l) in enumerate(FIX)}
            for fu in as_completed(futs):
                i = futs[fu]
                try:
                    ok, err = fu.result()
                except Exception as e:
                    ok, err = False, f"POOL-EXC:{type(e).__name__}"
                lab, exp = FIX[i][3], FIX[i][2]
                out[i] = (lab, ok, exp, ok == exp, str(err)[:40])
    except Exception as e:
        return f"POOL-INFRA-FAIL: {type(e).__name__}: {e}"
    return out

if __name__ == "__main__":
    print("=== SEQUENTIAL ===");
    sq = seq_run()
    for r in sq: print(f"  {r[0]:24} ok={r[1]} exp={r[2]} match={r[3]} err={r[4]}")
    print(f"  sequential correct-detection: {sum(1 for r in sq if r[3])}/{len(sq)}")
    print("=== POOLED ===");
    pl = pooled_run()
    if isinstance(pl, str): print(" ", pl)
    else:
        for r in pl: print(f"  {r[0]:24} ok={r[1]} exp={r[2]} match={r[3]} err={r[4]}")
        print(f"  pooled correct-detection: {sum(1 for r in pl if r and r[3])}/{len(pl)}")
    print("VERDICT: if SEQUENTIAL matches but POOLED all-False/errors -> use sequential scoring in harvest/reward.")
