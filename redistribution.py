"""
===============================================================================
 PHC-PULSE - REDISTRIBUTION OPTIMIZER (PuLP)
===============================================================================

WHAT THIS MODULE DOES
---------------------
Solves the "who ships medicine to whom" problem (Capability 4 of the project):
some PHCs have surplus stock, others have a shortfall, and we want the plan
that covers every shortfall while keeping transport distance low.

THE OPTIMIZATION IDEA, IN PLAIN WORDS
-------------------------------------
This is the classic "transportation problem", a type of Linear Program (LP):
  - Decision variables : the unknowns the solver picks - here, how many units
                         to ship from every surplus PHC to every shortfall PHC.
  - Objective function : the single number we minimize - total
                         (distance x units shipped).
  - Constraints        : the rules - a donor can't ship more than it has,
                         a receiver gets at most what it needs.

THE ONE NON-OBVIOUS TRICK (important!)
--------------------------------------
If we ONLY minimized distance, the solver would cheat: shipping zero units
costs zero, so the "optimal" plan is to do nothing. To force real coverage we
add one extra variable per receiver, `unmet`, representing units that did NOT
arrive, and we punish it with a huge cost BIG_M in the objective:

    minimize  (distance x shipped)  +  BIG_M x (total unmet)

Because BIG_M is bigger than ANY possible transport cost, the solver will
always prefer covering one more unit of need over saving any amount of
distance. Coverage first, efficiency second. When total shortfall exceeds
total surplus, this same mechanism automatically produces the best PARTIAL
plan (minimum unavoidable unmet) instead of crashing.

SOLVER NOTE (PuLP 4.0)
----------------------
PuLP no longer bundles the CBC solver. We point it at Homebrew's CBC when it
exists (typical on a Mac dev machine), and otherwise let PuLP find CBC on
the PATH (typical on Linux servers). Requires CBC to be installed somewhere:
    brew install cbc            (macOS)
    or: python -m pip install "pulp[cbc]"

HOW TO RUN
----------
    pip install "pulp[cbc]"   (inside your venv, once)
    python redistribution.py  (runs the built-in example below)
===============================================================================
"""

import os                          # for the os.path.exists() solver check

from pulp import (LpProblem, LpMinimize, LpInteger,
                  lpSum, LpSolveStatus, COIN_CMD)

# Path where Homebrew installs CBC on macOS. If it exists we use it
# explicitly; otherwise we let PuLP auto-discover CBC on the PATH.
CBC_PATH = "/opt/homebrew/opt/cbc/bin/cbc"


# =============================================================================
# SECTION 1: INPUT VALIDATION
# Optimization solvers give nonsense (or crash) on bad input, so we check
# everything BEFORE solving: whole-number, non-negative stocks, and a distance
# available for every donor-receiver pair the plan could use.
# =============================================================================

def _validate_inputs(surplus, shortfall, distance_matrix):
    """Raise a clear error if the inputs can't form a valid problem."""
    for label, d in (("surplus", surplus), ("shortfall", shortfall)):
        for phc, qty in d.items():
            if qty < 0:
                raise ValueError(f"{label}[{phc}] is negative ({qty}) - stock can't be negative.")
            if int(qty) != qty:
                raise ValueError(f"{label}[{phc}] must be a whole number, got {qty}.")

    for s in surplus:
        for t in shortfall:
            if s not in distance_matrix or t not in distance_matrix[s]:
                raise ValueError(f"distance_matrix is missing the distance for {s} -> {t}.")

    for s in surplus:
        for t in shortfall:
            dist = distance_matrix[s][t]
            if dist < 0:
                raise ValueError(f"Distance {s} -> {t} is negative ({dist}).")


# =============================================================================
# SECTION 2: THE SOLVER FUNCTION
# =============================================================================

def _pick_solver():
    """
    Returns the CBC solver object to use.

    PuLP 4.0 stopped bundling CBC, so we tell it where to find one:
      - On a Mac with Homebrew CBC installed, use that exact binary.
      - Anywhere else (Linux server, teammate's machine with cbcbox),
        let PuLP look for CBC on the PATH by itself.
    """
    if os.path.exists(CBC_PATH):
        return COIN_CMD(path=CBC_PATH, msg=False)
    return COIN_CMD(msg=False)


def solve_redistribution(surplus, shortfall, distance_matrix):
    """
    Compute the best medicine transfer plan between PHCs.

    Args:
        surplus         : dict {phc_id: extra units on the shelf}
        shortfall       : dict {phc_id: units this PHC still needs}
        distance_matrix : nested dict {donor: {receiver: km}} for every pair

    Returns:
        list of {'from', 'to', 'quantity'} dicts - one per active route,
        pairs with zero units omitted. Units are whole numbers.

    Behaviour when need exceeds supply: does NOT crash. It returns the plan
    that covers as much shortfall as possible (and prints a warning naming
    the minimum unavoidable unmet units).
    """
    _validate_inputs(surplus, shortfall, distance_matrix)

    # Trivial cases: nothing to give or nothing needed -> empty plan.
    if not surplus or not shortfall:
        print("Redistribution: nothing to do (surplus or shortfall is empty).")
        return []

    total_surplus = int(sum(surplus.values()))
    total_shortfall = int(sum(shortfall.values()))

    # --- Honest warning BEFORE solving: how much unmet need is unavoidable? ---
    if total_shortfall > total_surplus:
        unavoidable = total_shortfall - total_surplus
        print(f"WARNING: total shortfall ({total_shortfall} units) exceeds total "
              f"surplus ({total_surplus} units). At least {unavoidable} units "
              f"cannot be covered - returning the best partial plan.")

    # --- BIG_M: a penalty so big that covering need always beats saving km.
    # Total transport cost can never exceed max_distance x total_shortfall
    # (you can't ship more than the total need), so adding 1 makes BIG_M
    # strictly bigger than any possible distance cost. ---
    max_distance = max(distance_matrix[s][t] for s in surplus for t in shortfall)
    BIG_M = max_distance * total_shortfall + 1

    # --- Build the LP problem object ---
    prob = LpProblem("phc_redistribution", LpMinimize)

    # --- Decision variables ---
    # One integer variable per (donor, receiver) route: units shipped on it.
    # (PuLP 4.0: variables are created BY the problem via prob.add_variable();
    # calling LpVariable directly is no longer supported.)
    ship = {
        (s, t): prob.add_variable(f"ship_{s}_{t}", lowBound=0, cat=LpInteger)
        for s in surplus for t in shortfall
    }
    # One integer variable per receiver: units that go UNdelivered.
    unmet = {
        t: prob.add_variable(f"unmet_{t}", lowBound=0, cat=LpInteger)
        for t in shortfall
    }

    # --- Objective: total distance-cost + huge penalty for unmet need ---
    prob += (
        lpSum(distance_matrix[s][t] * ship[(s, t)] for s in surplus for t in shortfall)
        + BIG_M * lpSum(unmet.values())
    ), "total_cost"

    # --- Constraints ---
    # 1) A donor can't ship more than it actually has.
    for s in surplus:
        prob += lpSum(ship[(s, t)] for t in shortfall) <= surplus[s], f"supply_{s}"

    # 2) Every receiver ends up with (shipped + unmet) EXACTLY its shortfall.
    #    'unmet >= 0' above guarantees it never receives more than it needs.
    for t in shortfall:
        prob += lpSum(ship[(s, t)] for s in surplus) + unmet[t] == shortfall[t], f"demand_{t}"

    # --- Solve (exactly once, with the explicitly chosen solver) ---
    stats = prob.solve(_pick_solver())

    if stats.status != LpSolveStatus.Optimal:
        # Should not happen (the all-unmet plan is always feasible), but we
        # fail loudly instead of silently returning a broken plan.
        raise RuntimeError(f"Solver failed with status: {stats.status_str}")

    # --- Extract the plan: keep only routes that actually carry units ---
    plan = [
        {"from": s, "to": t, "quantity": int(round(ship[(s, t)].value()))}
        for s in surplus for t in shortfall
        if ship[(s, t)].value() is not None and ship[(s, t)].value() > 0.5
    ]
    return plan


# =============================================================================
# SECTION 3: BUILT-IN EXAMPLE (hardcoded, no files needed)
# 5 PHCs: three with surplus, two short. Distances in km between them.
# =============================================================================

if __name__ == "__main__":
    # Who has extra stock and how much.
    surplus = {"PHC_1": 100, "PHC_3": 50, "PHC_4": 30}

    # Who is short and how much they need. (Total need 140 < total surplus 180,
    # so a full solution exists.)
    shortfall = {"PHC_2": 80, "PHC_5": 60}

    # Road distance in km from every donor to every receiver.
    distance_matrix = {
        "PHC_1": {"PHC_2": 25,  "PHC_5": 90},
        "PHC_3": {"PHC_2": 60,  "PHC_5": 30},
        "PHC_4": {"PHC_2": 120, "PHC_5": 55},
    }

    print("=" * 60)
    print(" TEST 1: normal case (enough surplus to cover everything)")
    print("=" * 60)
    plan = solve_redistribution(surplus, shortfall, distance_matrix)

    print("\nRecommended transfer plan:")
    for move in plan:
        km = distance_matrix[move["from"]][move["to"]]
        print(f"  {move['from']} -> {move['to']}: {move['quantity']:>4} units   ({km} km)")
    print(f"  Total transfers: {len(plan)}, "
          f"total units moved: {sum(m['quantity'] for m in plan)}")

    # --- Bonus: show the graceful infeasibility path (requirement 4) ---
    print()
    print("=" * 60)
    print(" TEST 2: shortfall EXCEEDS surplus (must warn, not crash)")
    print("=" * 60)
    big_shortfall = {"PHC_2": 80, "PHC_5": 60, "PHC_9": 90}   # need 230, have 180
    plan2 = solve_redistribution(surplus, big_shortfall, {
        "PHC_1": {"PHC_2": 25, "PHC_5": 90, "PHC_9": 70},
        "PHC_3": {"PHC_2": 60, "PHC_5": 30, "PHC_9": 45},
        "PHC_4": {"PHC_2": 120, "PHC_5": 55, "PHC_9": 80},
    })
    print("\nBest partial plan:")
    for move in plan2:
        print(f"  {move['from']} -> {move['to']}: {move['quantity']:>4} units")
