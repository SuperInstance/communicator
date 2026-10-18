# Collaboration — multiple agents, one working tree

This repo is routinely worked by more than one agent at a time (ZCode and a
Claude instance, at minimum, as of 2026-10-18). The user's standing guidance:
*"you are one of many working — look and learn as you go,"* and *push often
with comments other agents and people can understand.*

## What actually happened (kept as the example, not the exception)

The Claude agent fixed two real bugs in freshly-written ZCode code before
ZCode had run it (a dict-iteration AttributeError in question_health's
first execution; a MidiEvent data-shape TypeError in score.py's first
execution), committed ZCode's uncommitted in-flight files into coherent
commits with honest messages, and shipped a presence-gate design that
superseded the rewrite ZCode was about to attempt. ZCode dropped its
version, added the complementary analysis (safe-parked verdicts, by-
construction calibration, the live end-to-end demo, README), and fixed two
never-executed bugs in the Claude agent's `render()` in return. Both
directions, net positive, no lost work.

## The protocol that made that work

1. **`git log` before you edit.** Commits minutes old are normal. Re-read
   any file you are about to change — the tree moves under you.
2. **Claim lanes, don't race them.** `.scratch/PHASE3_COORDINATION.md`
   (gitignored, local to this box) is the live claims file: take a unit,
   edit the file, never silently duplicate. Unclaimed work is fair game.
3. **Write commit messages for agents.** State what was measured, what the
   numbers were, what failed, and what the reader should not redo. The
   fleet's best messages read like lab notes, not changelogs.
4. **When superseded mid-flight, pivot to the complement.** If another
   agent's design beats the one you were building, drop yours and build the
   piece it doesn't cover. Duplicated effort is the only unrecoverable loss
   here.
5. **Commit green leftovers.** If another agent leaves finished, suite-green
   work uncommitted, commit it — with a message that says whose it was and
   what you verified. Uncommitted work on a shared tree is one crash from
   gone.
6. **Verify your negative controls.** A sabotage test whose patch matched
   nothing once reported "25 passed" and nearly got recorded as a blind
   test. Assert the sabotage actually applied before recording it.
7. **A surprise is more likely a colleague than corruption.** Files
   changing between your read and your write, suite counts moving between
   runs — check `git log` before concluding anything is broken.
