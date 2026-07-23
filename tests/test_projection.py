# CPU unit tests for the hard-projection control-flow/math (no torch model needed).
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_training.projection import (
    floor_logp, violations, max_violation, batches, should_project,
    projection_loop_plan, ProjectionConfig, ProjectionState,
)


def test_floor_logp():
    assert abs(floor_logp(-5.0, 0.5) - (-5.0 + math.log(0.5))) < 1e-9
    assert floor_logp(-5.0, 1.0) == -5.0            # alpha=1 -> floor == ref
    try:
        floor_logp(-5.0, 0.0); assert False
    except ValueError:
        pass


def test_violations_strictly_below_only():
    ref = [-5.0, -5.0, -5.0]
    # alpha=1 -> floor=-5: below, exactly-at, above
    pol = [-6.0, -5.0, -4.0]
    assert violations(pol, ref, 1.0) == [0]         # only the strictly-below one
    # exactly-at floor is feasible (not selected)
    assert 1 not in violations(pol, ref, 1.0)


def test_max_violation_zero_when_feasible():
    ref = [-5.0, -3.0]; pol = [-5.0, -2.0]
    assert max_violation(pol, ref, 1.0) == 0.0
    # one violation of 1 nat
    assert abs(max_violation([-6.0, -2.0], ref, 1.0) - 1.0) < 1e-9


def test_alpha_slack_reduces_violations():
    ref = [-5.0]; pol = [-5.5]
    assert violations(pol, ref, 1.0) == [0]         # floor -5: violated
    assert violations(pol, ref, 0.5) == []          # floor -5.693: satisfied


def test_batches_cover_all_indices_in_order():
    idx = list(range(10))
    got = list(batches(idx, 4))
    assert got == [[0,1,2,3],[4,5,6,7],[8,9]]
    assert sum(len(b) for b in got) == 10


def test_should_project_amortization():
    c = ProjectionConfig(every=1)
    assert all(should_project(s, c) for s in range(5))
    c = ProjectionConfig(every=3)
    assert [should_project(s, c) for s in range(6)] == [True, False, False, True, False, False]


def test_loop_terminates_at_max_steps_if_never_feasible():
    # policy that NEVER improves -> loop must stop at max_steps (bounded cost guarantee).
    ref = [-5.0, -5.0]
    cfg = ProjectionConfig(alpha=1.0, max_steps=3, batch_size=8)
    plan = projection_loop_plan(lambda: [-9.0, -9.0], ref, cfg)
    assert len(plan) == 3                            # exactly capped
    for sub, bts in plan:
        assert bts == [[0, 1]]                       # both traces violating, one batch


def test_loop_early_stops_when_feasible():
    # mock policy that becomes feasible after 2 reads (simulates correction working).
    ref = [-5.0]
    seq = iter([[-9.0], [-9.0], [-4.0]])             # 3rd read is feasible
    cfg = ProjectionConfig(alpha=1.0, max_steps=5, batch_size=8)
    plan = projection_loop_plan(lambda: next(seq), ref, cfg)
    # sub0 reads -9 (violate, plan), sub1 reads -9 (violate, plan), sub2 reads -4 (feasible, stop)
    assert len(plan) == 2


def test_loop_no_op_when_already_feasible():
    ref = [-5.0, -3.0]
    cfg = ProjectionConfig(alpha=1.0, max_steps=5)
    plan = projection_loop_plan(lambda: [-4.0, -2.0], ref, cfg)
    assert plan == []                                # nothing to correct


def test_state_records_history():
    st = ProjectionState()
    st.step = 10; st.record(max_v=0.3, n_v=2, corr_steps=1)
    st.step = 20; st.record(max_v=0.0, n_v=0, corr_steps=0)
    assert st.total_correction_steps == 1
    assert st.last_max_violation == 0.0 and st.last_n_violations == 0
    assert [h["step"] for h in st.history] == [10, 20]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL PROJECTION TESTS PASSED")
