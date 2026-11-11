#!/usr/bin/env python3
"""Unit tests for the Project 2 temporal state machine.

Run: python3 -m unittest test_state_machine -v   (stdlib only)

Sequences are synthetic frame-label streams at 2 fps unless noted -
they test the *judging logic*, not the classifier. Real-video sequences
live in video-sequences/ and are exercised by sequence_evaluate.py.
"""
import unittest

from state_machine import (CommandSession, FrameObservation, SessionConfig,
                           TemporalSmoother, canonical_label, run_sequence)

FPS_DT = 0.5  # 2 fps sampling, as in the app design


def stream(labels, dt=FPS_DT, conf=0.95):
    return [FrameObservation(t=i * dt, label=x, confidence=conf)
            for i, x in enumerate(labels)]


def cfg(**kw):
    base = dict(target_duration_s=5.0, smooth_window=3, settle_grace_s=2.0,
                break_confirm_s=1.0, uncertain_timeout_s=3.0)
    base.update(kw)
    return SessionConfig(**base)


class TestCanonical(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(canonical_label("Sit"), "sitting")
        self.assertEqual(canonical_label("lying down"), "lying_down")
        self.assertEqual(canonical_label("DOWN"), "lying_down")
        self.assertEqual(canonical_label("Walk"), "walking")
        self.assertEqual(canonical_label("gibberish"), "uncertain_frame")
        self.assertEqual(canonical_label(None), "uncertain_frame")


class TestSmoother(unittest.TestCase):
    def test_majority_vote(self):
        s = TemporalSmoother(window=3)
        out = [s.push(FrameObservation(0, x, 0.9)) for x in
               ["sitting", "standing", "sitting", "sitting"]]
        self.assertEqual(out[0], "sitting")
        self.assertEqual(out[-1], "sitting")

    def test_low_confidence_is_uncertain(self):
        s = TemporalSmoother(window=3, min_confidence=0.6)
        self.assertEqual(s.push(FrameObservation(0, "sitting", 0.3)),
                         "uncertain_frame")

    def test_no_dog_is_uncertain(self):
        s = TemporalSmoother(window=3)
        self.assertEqual(
            s.push(FrameObservation(0, "sitting", 0.99, dog_present=False)),
            "uncertain_frame")


class TestVerdicts(unittest.TestCase):
    def test_clean_success(self):
        v = run_sequence(stream(["sitting"] * 14), "sitting", cfg())
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertGreaterEqual(v.held_s, 5.0)

    def test_early_break_proposal_example(self):
        # Proposal section 6 example: Stay, sit 0-2s, then stand/walk away.
        labels = ["sitting"] * 5 + ["standing"] * 3 + ["walking"] * 6
        v = run_sequence(stream(labels), "sitting", cfg())
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertIsNotNone(v.break_at_s)
        self.assertLess(v.held_s, 5.0)

    def test_single_noisy_frame_does_not_break(self):
        labels = ["sitting"] * 14
        labels[7] = "standing"          # one-frame glitch mid-hold
        v = run_sequence(stream(labels), "sitting", cfg())
        self.assertEqual(v.verdict, "SUCCESS")

    def test_down_success(self):
        v = run_sequence(stream(["lying down"] * 14), "lying_down", cfg())
        self.assertEqual(v.verdict, "SUCCESS")

    def test_walking_success(self):
        v = run_sequence(stream(["walking"] * 16), "walking",
                         cfg(target_duration_s=6.0))
        self.assertEqual(v.verdict, "SUCCESS")

    def test_target_never_reached_is_uncertain(self):
        v = run_sequence(stream(["standing"] * 10), "sitting", cfg())
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_low_confidence_run_is_uncertain(self):
        obs = [FrameObservation(t=i * FPS_DT, label="sitting", confidence=0.2)
               for i in range(14)]
        v = run_sequence(obs, "sitting", cfg())
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_no_dog_run_is_uncertain(self):
        obs = [FrameObservation(t=i * FPS_DT, label=None, dog_present=False)
               for i in range(14)]
        v = run_sequence(obs, "sitting", cfg())
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_interval_ending_early_is_uncertain_not_success(self):
        v = run_sequence(stream(["sitting"] * 6), "sitting", cfg())  # 3s < 5s
        self.assertEqual(v.verdict, "UNCERTAIN")
        self.assertIn("incomplete", v.reason)

    def test_lost_dog_mid_hold_is_uncertain(self):
        obs = stream(["sitting"] * 4)
        obs += [FrameObservation(t=2.0 + i * FPS_DT, label=None,
                                 dog_present=False) for i in range(8)]
        v = run_sequence(obs, "sitting", cfg())
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_short_uncertain_gap_does_not_ruin_hold(self):
        obs = stream(["sitting"] * 4)
        obs += [FrameObservation(t=2.0, label=None, confidence=0.1),
                FrameObservation(t=2.5, label=None, confidence=0.1)]
        obs += stream(["sitting"] * 10, dt=FPS_DT)[2:]
        for i, o in enumerate(obs):
            o.t = i * FPS_DT
        v = run_sequence(obs, "sitting", cfg())
        self.assertEqual(v.verdict, "SUCCESS")

    def test_success_locks_before_late_break(self):
        labels = ["sitting"] * 14 + ["standing"] * 8
        s = CommandSession("sitting", cfg())
        v = None
        for o in stream(labels):
            v = s.feed(o)
            if v is not None:
                break
        self.assertEqual(v.verdict, "SUCCESS")

    def test_settle_within_grace(self):
        # dog takes 1.5s to sit after the command, then holds: still SUCCESS
        labels = ["standing"] * 3 + ["sitting"] * 14
        v = run_sequence(stream(labels), "sitting", cfg())
        self.assertEqual(v.verdict, "SUCCESS")
        # raw labels turn sitting at t=1.5; the 3-frame majority smoother
        # confirms it one frame later, so the hold clock starts at t=2.0
        self.assertAlmostEqual(v.target_reached_at_s, 2.0)


if __name__ == "__main__":
    unittest.main()
