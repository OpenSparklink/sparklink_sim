"""Bounded live fine-phase continuations on the word fabric (§6.2.3).

This consumes operands only as generators issue them. It never records a
future instruction/answer trace. Generator objects represent compiled finite
contexts whose register reservation is separately charged in the score cost.
"""

from nearlink_sdr.sim.word_arithmetic import WordCommand

__all__ = ["ScoreWordRuntime", "score_word_command"]


def score_word_command(op, tag_id):
    """Static node formats; no sign inference from a runtime operand value."""
    code, tag, bits = op["code"], op["tag"], op["bits"]
    wa = wb = bits
    sa = sb = op["signed"]
    if code == "tile":
        wa, wb, sa, sb = 24, 17, False, False
    elif code == "rne":
        wa, wb, sa, sb = 33, 4, True, False
    elif ".abs_" in tag:
        sa = not tag.startswith(("energy_product.", "rank_left.", "rank_right."))
        sb = False
    elif tag.endswith(".sign"):
        wa, sa, sb = bits - 1, False, False
    elif tag == "geometry_cmp":
        wa = wb = 17
        sa = sb = True
    return WordCommand(code, op["a"], op["b"], wa, wb, sa, sb, bits, op["signed"], tag_id)


class ScoreWordRuntime:
    """At most9 live generators,one pending command and one mailbox per phase."""

    def __init__(self, fabric, advance, issue, complete, tags):
        self.fabric, self.advance = fabric, advance
        self.issue_callback, self.complete_callback = issue, complete
        self.tags = {tag: k for k, tag in enumerate(sorted(tags))}
        self.peak_live = 0

    def fork(self, generators):
        """Join one symbol across phases; no retained full-symbol vectors."""
        if not isinstance(generators, tuple) or len(generators) > self.fabric.a.contexts:
            raise ValueError("Need at most the compiled number of live contexts")
        n = len(generators)
        self.peak_live = max(self.peak_live, n)
        pending, in_flight, done, results = [None] * n, [None] * n, [False] * n, [None] * n

        def resume(k, value=None, initial=False):
            try:
                pending[k] = next(generators[k]) if initial else generators[k].send(value)
            except StopIteration as finished:
                done[k], results[k], pending[k] = True, finished.value, None

        for k in range(n):
            resume(k, initial=True)
        issue_head = take_head = 0
        while not all(done):
            for offset in range(n):
                k = (take_head + offset) % n
                if in_flight[k] is None:
                    continue
                reply = self.fabric.take(k)
                if reply is not None:
                    op, token = in_flight[k]
                    if reply.token != token or reply.tag != self.tags[op["tag"]]:
                        raise RuntimeError("Stale or misrouted word result")
                    self.complete_callback(op, reply.value, reply.overflow)
                    in_flight[k] = None
                    resume(k, reply.value)
                    take_head = (k + 1) % n
                    break
            for offset in range(n):
                k = (issue_head + offset) % n
                if done[k] or in_flight[k] is not None:
                    continue
                op = pending[k]
                typed = score_word_command(op, self.tags[op["tag"]])
                token = self.fabric.next_token[k]
                if self.fabric.issue(typed, k, token):
                    self.issue_callback(op, None)
                    in_flight[k], pending[k] = (op, token), None
                    issue_head = (k + 1) % n
                    break
            if not all(done):
                self.advance()
        return results

    def execute(self, command):
        """Use the same finite engine for scalar ranking/replay operations."""

        def scalar():
            return (yield command)

        return self.fork((scalar(),))[0]
