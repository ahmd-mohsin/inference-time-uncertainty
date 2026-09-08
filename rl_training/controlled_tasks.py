# Controlled EXECUTABLE task families for the §50 award-target program (H-A consolidation-interval,
# H-C coverage, H-D structured verification, frozen compositional demo). Ground truth + operation labels
# come from an executable generator (NOT an LLM), so "same operation / new composition" splits are exact.
#
# A task = a short straight-line program over integers: a sequence of labeled ops applied left-to-right to a
# running accumulator (seeded from the first operand). Answer = executed result (independent verifier).
# Ops are the intermediate "reasoning transitions" H-C measures coverage over.
#
# Splits per base family (kept SEPARATE — a paraphrase is NOT a compositional-generalization test):
#   original     : the exact seed instance
#   surface      : same op-sequence, same operands, reworded prompt (paraphrase)
#   same_op      : same op-sequence, NEW operands (new instance of the same operations)
#   new_compose  : same op MULTISET in a DIFFERENT order/position (composition generalization)
#   missing_op   : uses an op never supplied in training (boundary split; expect failure)
#
# Usage:
#   from rl_training.controlled_tasks import gen_family, verify, OPS
#   fam = gen_family(seed=0, n_ops=3)   # -> dict with 'original','surface','same_op','new_compose'
import argparse, json, random, re

# executable ops on (acc:int, operand:int) -> int. Labels are the coverage units.
OPS = {
    "add":  lambda a, b: a + b,
    "sub":  lambda a, b: a - b,
    "mul":  lambda a, b: a * b,
    "mod":  lambda a, b: a % b if b != 0 else a,
    "max":  lambda a, b: max(a, b),
    "min":  lambda a, b: min(a, b),
    "absdiff": lambda a, b: abs(a - b),
}
OP_NAMES = list(OPS)
_PHRASE = {"add": "add {b}", "sub": "subtract {b}", "mul": "multiply by {b}",
           "mod": "take the remainder when divided by {b}", "max": "take the maximum with {b}",
           "min": "take the minimum with {b}", "absdiff": "take the absolute difference with {b}"}
_PHRASE2 = {"add": "increase it by {b}", "sub": "decrease it by {b}", "mul": "scale it by a factor of {b}",
            "mod": "replace it with its remainder modulo {b}", "max": "keep the larger of it and {b}",
            "min": "keep the smaller of it and {b}", "absdiff": "replace it with |it − {b}|"}

def _execute(start, prog):
    acc = start
    for op, b in prog: acc = OPS[op](acc, b)
    return acc

def _render(start, prog, phrasebook):
    steps = "; then ".join(phrasebook[op].format(b=b) for op, b in prog)
    return (f"Start with {start}; then {steps}. Show brief reasoning and put the final integer answer "
            f"in \\boxed{{}}.")

def _rand_prog(rng, ops, n):
    return [(op, rng.randint(2, 12)) for op in rng.sample(ops, k=1) * 0 + [rng.choice(ops) for _ in range(n)]]

def gen_family(seed, n_ops=3, allowed_ops=None):
    """One base family with its 4 in-support splits. allowed_ops restricts the op vocabulary (for coverage)."""
    rng = random.Random(seed)
    ops = allowed_ops or OP_NAMES
    start = rng.randint(2, 20)
    prog = [(rng.choice(ops), rng.randint(2, 12)) for _ in range(n_ops)]
    def mk(s, p, phr): return {"prompt": _render(s, p, phr), "gold": str(_execute(s, p)),
                               "ops": sorted(set(o for o, _ in p)), "prog": p, "start": s}
    original = mk(start, prog, _PHRASE)
    surface = mk(start, prog, _PHRASE2)                                   # same compute, reworded
    s2 = rng.randint(2, 20); p2 = [(o, rng.randint(2, 12)) for o, _ in prog]
    same_op = mk(s2, p2, _PHRASE)                                         # same ops, new operands
    p3 = prog[::-1] if len(prog) > 1 else prog                            # same op multiset, new order
    if p3 == prog and len(prog) > 1: p3 = [prog[1], prog[0]] + prog[2:]
    new_compose = mk(rng.randint(2, 20), p3, _PHRASE)
    return {"family": seed, "n_ops": n_ops, "allowed_ops": ops,
            "original": original, "surface": surface, "same_op": same_op, "new_compose": new_compose}

def missing_op_family(seed, held_out_op, n_ops=3):
    """Boundary split: a family that REQUIRES an op never supplied in training."""
    rng = random.Random(10_000 + seed)
    start = rng.randint(2, 20)
    others = [o for o in OP_NAMES if o != held_out_op]
    prog = [(held_out_op, rng.randint(2, 12))] + [(rng.choice(others), rng.randint(2, 12)) for _ in range(n_ops - 1)]
    rng.shuffle(prog)
    return {"prompt": _render(start, prog, _PHRASE), "gold": str(_execute(start, prog)),
            "ops": sorted(set(o for o, _ in prog)), "prog": prog, "start": start, "held_out_op": held_out_op}

_BOX = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
def verify(text, gold):
    """Independent exact-integer verifier (executable ground truth)."""
    if text is None: return False
    m = _BOX.findall(text)
    if m: return m[-1].strip() == str(gold).strip()
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return bool(nums) and nums[-1] == str(gold).strip()

def build(out, n_families, n_ops, split, allowed_ops=None, seed0=0):
    rng_ops = allowed_ops.split(",") if allowed_ops else None
    rows = []
    for i in range(n_families):
        fam = gen_family(seed0 + i, n_ops=n_ops, allowed_ops=rng_ops)
        r = fam[split]
        rows.append({"prompt": r["prompt"], "gold": r["gold"], "ops": r["ops"], "family": fam["family"], "split": split})
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"[controlled_tasks] wrote {len(rows)} {split} tasks (n_ops={n_ops}, ops={rng_ops or 'all'}) -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-ops", type=int, default=3)
    ap.add_argument("--split", default="original", choices=["original", "surface", "same_op", "new_compose"])
    ap.add_argument("--allowed-ops", default=""); ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.n, a.n_ops, a.split, a.allowed_ops or None, a.seed0)
