"""Week 2 studio — Task 1 starter (complete the agent taxonomy).

Fill in the two agents below, then run ``python3 test_agents.py``. All required
tests must pass, INCLUDING the ones that assert a *surprising* property:

  - the simple reflex agent PROVABLY fails on a world the model-based agent
    handles (partial observability) — watch a guarantee fail;
  - the utility agent RATIONALLY leaves a room dirty on at least one config —
    not a bug, but specification-following under a cost model.

You are given the world and two agents in ``vacuum.py`` (simple_reflex,
model_based). You write ``goal_based`` and ``utility_based`` here.
"""
from vacuum import VacuumWorld, run, simple_reflex, model_based


# ---- Task 1: complete the taxonomy ------------------------------------------

def goal_based(goal=(False, False)):
    """Return an agent that PLANS toward an explicit goal, then STOPS.

    The goal is a target dirt-state; ``(False, False)`` means "both rooms
    clean". Unlike the reflex agent (which oscillates forever) the goal-based
    agent must recognise the goal is reached and return ``"NoOp"`` from then on.

    Suggested internal state (like model_based): remember which rooms you have
    cleaned and where you are, since the percept only shows the current room.

    Returns: a function ``agent(percept) -> action``.
    """
    # Internal state: what we believe about each room. None = never perceived,
    # True = dirty, False = clean. The percept only ever tells us about the room
    # we are standing in, so the belief is how the goal test becomes possible.
    belief = [None, None]

    def goal_reached():
        return all(b is not None and b == g for b, g in zip(belief, goal))

    def agent(percept):
        loc, dirty = percept
        belief[loc] = dirty                     # perceive: update the model

        if goal_reached():
            return "NoOp"                       # plan is finished — stop acting

        if dirty and not goal[loc]:
            belief[loc] = False                 # Suck will make it clean
            return "Suck"

        # This room already matches the goal; the only room that can still be
        # wrong is the other one, so head there.
        other = 1 - loc
        return "Right" if other == 1 else "Left"

    return agent


def utility_based(move_cost=1, suck_cost=2, clean_reward=3):
    """Return an agent that maximises NET utility one step at a time.

    Cost model (same numbers as ``tournament.py``): each move costs
    ``move_cost``, each Suck costs ``suck_cost``, and each clean room earns
    ``clean_reward`` per step. The agent function only sees ``percept`` — it has
    no lookahead — so it must reason *greedily* about the immediate net change.

    The intended discovery: a greedy utility agent SOMETIMES LEAVES A ROOM
    DIRTY, because paying to travel to a room it cannot see the payoff of is not
    immediately worth it. Report it as a result, not a bug — this is the first
    time an agent makes a decision you did not intend but cannot argue with.

    Returns: a function ``agent(percept) -> action``.
    """
    # Same belief state as the goal-based agent: None = never perceived.
    # (The notebook's `utility_agent` tracked `clean` instead; same idea, but it
    # never read `loc`, which is why it assumed it started in room 0.)
    belief = [None, None]

    def agent(percept):
        loc, dirty = percept
        belief[loc] = dirty

        # Sucking HERE: pay suck_cost once, and the reward lands on this very
        # step (world.step applies Suck before the reward is counted).
        if dirty and clean_reward - suck_cost > 0:
            belief[loc] = False
            return "Suck"

        # Crossing: the payoff sits in the OTHER room, which the percept does
        # not show. The cheapest way it could ever pay is move-then-suck, so
        # that is the most generous number a horizon-free agent may put on it:
        #     -move_cost - suck_cost + clean_reward
        # This is the notebook's `move_cost < 1.0` test, re-derived in
        # tournament.py's numbers (3 - 2 = 1, so again the threshold is 1).
        other = 1 - loc
        if belief[other] is not False and clean_reward - suck_cost - move_cost > 0:
            return "Right" if other == 1 else "Left"

        return "NoOp"          # crossing is not worth it

    return agent


# ---- Task 2 sketch (the LLM as agent function) ------------------------------
# See README.md §"Task 2". Drop a language model into the SAME percept->action
# socket. You supply the model call; you must handle malformed output and LOG
# every parse failure (it belongs on the scorecard — axis 8).
#
#   from aicourse.llm import LLM
#   model = LLM()  # your configured local backend
#
#   PROMPT = """You control a vacuum robot in a 2-room world.
#   Rooms 0 and 1. Each is Clean or Dirty.
#   Actions: Left, Right, Suck, NoOp.
#   History of (percept, action) so far:
#   {history}
#   Current percept: location={loc}, status={status}
#   Reply with exactly one action word and nothing else."""
#
#   def parse_action(raw):
#       # must survive "Suck.", "I would suck", or a paragraph of reasoning.
#       ...  # return one of Left/Right/Suck/NoOp, or None on failure (log it!)
#
#   def llm_agent(history):
#       def agent(percept):
#           loc, dirty = percept
#           resp = model.complete(PROMPT.format(history=fmt(history),
#                                               loc=loc,
#                                               status="Dirty" if dirty else "Clean"))
#           return parse_action(resp.text)
#       return agent
#
# Run the same 8 configs; run ONE config 5x and count distinct action sequences
# (reproducibility — axis 5, made visceral). Compare to the four classical agents.


if __name__ == "__main__":
    # Quick smoke test: run each agent you've filled in on one config.
    from tournament import tournament, net_utility

    factories = {
        "reflex": lambda: simple_reflex,
        "model": model_based,
    }
    for name, make in [("goal", goal_based), ("utility", utility_based)]:
        try:
            make()  # will raise NotImplementedError until you fill it in
            factories[name] = make
        except NotImplementedError:
            print(f"  {name:<10} not implemented yet")

    if len(factories) > 2:
        print()
        tournament(factories)
