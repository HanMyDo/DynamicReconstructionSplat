"""The depth veto cuts background that SAM fused into a mover's object.

THE CASE, taken from the measured one (TUM halfsphere frame 457): the detector
seeds a person, SAM propagates an object that covers the person AND the ceiling
behind them, and every existing rail waves it through -- the growth rail asks
"too big?" and a ceiling strip is not, while the seed-support gate is PER OBJECT
and the person's own seeds vouch for the whole thing. Seed area 0.0062 became a
mask over ~15% of the frame.

Depth is the one signal that separates them, and SAM is never shown it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.encoder.sam_video_complete import _depth_split_keep


def _scene():
    """A 'person' at 2 m and a 'ceiling' at 6 m, touching, in one mask."""
    H = W = 64
    depth = np.full((H, W), 6.0, np.float32)     # far background
    person = np.zeros((H, W), bool)
    person[30:60, 20:40] = True                  # a standing figure
    depth[person] = 2.0                          # much nearer
    ceiling = np.zeros((H, W), bool)
    ceiling[0:12, :] = True                      # a strip across the top
    fused = person | ceiling                     # what SAM returns as ONE object
    return depth, person, ceiling, fused


def test_veto_removes_the_unsupported_far_part():
    depth, person, ceiling, fused = _scene()
    support = np.zeros_like(person)
    support[40:50, 25:35] = True                 # seeds on the person only
    kept, dropped = _depth_split_keep(fused, depth, support, jump=0.08, frac=0.02)
    assert dropped >= 1, "the ceiling part should have been cut"
    # The person survives essentially intact, the ceiling is gone.
    assert (kept & person).sum() >= 0.9 * person.sum(), "the person was eroded"
    assert (kept & ceiling).sum() <= 0.02 * ceiling.sum(), "the ceiling survived"
    print(f"[1] person kept {(kept & person).sum()}/{person.sum()}, "
          f"ceiling kept {(kept & ceiling).sum()}/{ceiling.sum()}, parts dropped {dropped}")


def test_veto_keeps_a_supported_object_whole():
    """A mask with no depth jump inside it must come back unchanged, or the veto
    would quietly erode every honest mask it inspects."""
    depth, person, _, _ = _scene()
    support = np.zeros_like(person)
    support[40:50, 25:35] = True
    kept, dropped = _depth_split_keep(person, depth, support, jump=0.08, frac=0.02)
    assert dropped == 0
    assert kept.sum() == person.sum(), "an unsplit, supported object changed size"
    print(f"[2] unsplit object preserved exactly: {kept.sum()} px")


def test_veto_is_off_by_default_value():
    depth, person, _, fused = _scene()
    support = np.zeros_like(person)
    kept, dropped = _depth_split_keep(fused, depth, support, jump=0.0, frac=0.02)
    assert dropped == 0 and kept.sum() == fused.sum(), "jump=0 must be a no-op"
    print("[3] jump=0 is a no-op")


def test_both_parts_supported_survive():
    """Two depth-separated parts that are BOTH seeded are both kept -- the veto
    tests support, not nearness, so a person in front of a seeded moving door
    must not lose the door."""
    depth, person, ceiling, fused = _scene()
    support = np.zeros_like(person)
    support[40:50, 25:35] = True
    support[2:8, 20:40] = True                   # the far part is seeded too
    kept, dropped = _depth_split_keep(fused, depth, support, jump=0.08, frac=0.02)
    assert dropped == 0
    assert (kept & ceiling).sum() >= 0.9 * ceiling.sum()
    print("[4] a seeded far part is kept")


if __name__ == "__main__":
    test_veto_removes_the_unsupported_far_part()
    test_veto_keeps_a_supported_object_whole()
    test_veto_is_off_by_default_value()
    test_both_parts_supported_survive()
    print("OK")
