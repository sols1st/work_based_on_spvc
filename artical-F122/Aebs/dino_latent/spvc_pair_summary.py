"""Small dependency-free accumulators for matched old/new policy diagnostics."""

from collections import Counter
import math


class PairSummary:
    def __init__(self):
        self.n = 0
        self.sums = Counter()
        self.maxima = Counter()
        self.counts = Counter()
        self.outcomes = {name: Counter() for name in ("old_image", "old_q", "new_image", "new_q")}

    def add(self, actions, outcomes):
        if len(actions) != 4 or len(outcomes) != 4 or not all(math.isfinite(float(a)) for a in actions):
            raise ValueError("four finite actions and four outcomes required")
        self.n += 1
        for name, value in (
            ("old_dual_path", actions[0]-actions[1]),
            ("new_dual_path", actions[2]-actions[3]),
            ("image_policy_drift", actions[2]-actions[0]),
            ("q_policy_drift", actions[3]-actions[1]),
        ):
            self.sums[name] += abs(float(value))
            self.maxima[name] = max(self.maxima[name], abs(float(value)))
        for name, outcome in zip(self.outcomes, outcomes):
            self.outcomes[name][outcome] += 1
        for name, offset in (("old", 0), ("new", 2)):
            image_bad, q_bad = outcomes[offset] == "unsafe", outcomes[offset+1] == "unsafe"
            self.counts[name+"_image_only_next_unsafe"] += int(image_bad and not q_bad)
            self.counts[name+"_q_only_next_unsafe"] += int(q_bad and not image_bad)
            self.counts[name+"_both_next_unsafe"] += int(image_bad and q_bad)
            self.counts[name+"_outcome_disagreement"] += int(outcomes[offset] != outcomes[offset+1])
        for name, a, b in (("image", 0, 2), ("q", 1, 3)):
            self.counts[name+"_new_unsafe_old_not"] += int(outcomes[b] == "unsafe" and outcomes[a] != "unsafe")
            self.counts[name+"_old_unsafe_new_not"] += int(outcomes[a] == "unsafe" and outcomes[b] != "unsafe")

    def result(self):
        return {
            "checked_nonterminal_pairs": self.n,
            "action": {name: {"mae": self.sums[name]/self.n if self.n else None,
                              "max_abs": self.maxima[name] if self.n else None}
                       for name in ("old_dual_path", "new_dual_path", "image_policy_drift", "q_policy_drift")},
            "next_outcomes": {name: dict(counts) for name, counts in self.outcomes.items()},
            "counts": dict(self.counts),
        }
