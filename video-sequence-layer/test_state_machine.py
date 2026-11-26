#!/usr/bin/env python3
"""Unit tests for the Project 2 temporal state machine.

Run: python3 -m unittest test_state_machine -v   (stdlib only)
     (or python3 -m pytest -q test_state_machine.py if pytest is installed)

Sequences are synthetic frame-label streams - they test the *judging
logic*, not the classifier. Real-video sequences live in video-sequences/
and are exercised by sequence_evaluate.py.

Three groups:
  TestCanonical / TestSmoother / TestVerdicts - the original 17 tests
      (Nov 10-11). They still pass the Nov 11 numbers explicitly
      (smooth window 3, break confirm 1 s, lost 3 s) at 2 fps, so they
      now check the Nov 25 rules at those numbers. One expectation
      changed with the rules (marked "RULE CHANGE").
  TestAppRules - the 21 JUnit tests of the Android app's
      SequenceSession (SequenceSessionTest.java), ported one to one:
      app defaults, synthetic 5 fps streams (t rounded to 3 decimals as
      in the Java helper), Model A style confidences.
  TestConfigAndPresets - config_for_command / PRESETS, and the frozen
      Nov 11 module (state_machine_nov11.py).
"""
import math
import unittest

import state_machine_nov11
from state_machine import (PRESETS, UNCERTAIN_FRAME, CommandSession,
                           FrameObservation, SessionConfig, TemporalSmoother,
                           canonical_label, config_for_command,
                           is_reach_then_hold, run_sequence)

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
        # RULE CHANGE (Nov 25, reach time on the raw time base): raw
        # labels turn sitting at t=1.5 and the 3-frame majority smoother
        # confirms it one frame later (t=2.0). The Nov 11 rules started
        # the hold clock at the smoothed flip (expected 2.0); the app's
        # rules start it at the first raw target frame of the confirmed
        # run, so the expectation is now 1.5.
        self.assertAlmostEqual(v.target_reached_at_s, 1.5)



# ---------------------------------------------------------------------------
# Port of the Android app's SequenceSessionTest.java (21 tests, same names
# in snake_case, same streams, same expectations).

DT = 0.2  # synthetic 5 fps streams, as in the Java tests


def _java_round(x):
    """Math.round: half-up."""
    return math.floor(x + 0.5)


def stay(seconds):
    return CommandSession("sitting", SessionConfig(target_duration_s=seconds),
                          "Stay")


def sit():
    return CommandSession("sitting", SessionConfig(target_duration_s=5,
                                                   reach_then_hold=True),
                          "Sit")


def down():
    return CommandSession("lying_down", SessionConfig(target_duration_s=5,
                                                      reach_then_hold=True),
                          "Down")


def run(s, t, label, conf, dog, seconds, dt=DT):
    """Feed `label` from t[0] for `seconds` at interval dt; returns the
    verdict if one was decided and leaves the next free t in t[0]."""
    v = None
    n = _java_round(seconds / dt)
    i = 0
    while i < n and v is None:
        v = s.feed(FrameObservation(t=t[0], label=label, confidence=conf,
                                    dog_present=dog))
        t[0] = _java_round((t[0] + dt) * 1000) / 1000.0
        i += 1
    return v


class TestAppRules(unittest.TestCase):
    def test_stay_success_after_full_hold(self):
        s = stay(5)
        t = [0.0]
        v = run(s, t, "sitting", 0.9, True, 6)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertAlmostEqual(v.target_reached_at_s, 0.0, delta=1e-9)
        self.assertGreaterEqual(v.held_s, 5.0)

    def test_stay_break_is_timestamped_at_first_wrong_frame(self):
        s = stay(10)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 3.0))  # 0.0..2.8
        break_start = t[0]                                       # 3.0
        v = run(s, t, "standing", 0.9, True, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertAlmostEqual(v.break_at_s, break_start, delta=1e-9)
        self.assertAlmostEqual(v.held_s, 3.0, delta=1e-9)

    def test_one_glitch_frame_does_not_break_a_stay(self):
        s = stay(4)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 2.0))
        self.assertIsNone(run(s, t, "standing", 0.9, True, 0.2))  # 1 frame
        v = run(s, t, "sitting", 0.9, True, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")

    def test_sit_not_reached_in_three_seconds_is_early_break(self):
        s = sit()
        t = [0.0]
        v = run(s, t, "standing", 0.9, True, 4.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertIsNone(v.target_reached_at_s)
        self.assertAlmostEqual(v.held_s, 0.0, delta=1e-9)

    def test_sit_reached_then_held_is_success(self):
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 1.0))
        v = run(s, t, "sitting", 0.9, True, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertLess(v.target_reached_at_s, 3.0)

    def test_low_confidence_frames_count_as_unclear(self):
        s = stay(5)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 1.0))
        # below the 0.65 threshold for 2 s -> lost -> UNCERTAIN
        v = run(s, t, "sitting", 0.4, True, 2.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_no_dog_for_too_long_is_uncertain(self):
        s = stay(5)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 1.0))
        v = run(s, t, None, 0, False, 2.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")
        self.assertIn("lost", v.reason)

    def test_short_occlusion_is_tolerated(self):
        s = stay(4)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 1.0))
        self.assertIsNone(run(s, t, None, 0, False, 1.0))  # 1.0 s < 1.5 s
        v = run(s, t, "sitting", 0.9, True, 4.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")

    def test_stopping_early_is_uncertain_not_success(self):
        s = stay(10)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 3.0))
        v = s.finish()
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_readiness_smoother_respects_confidence(self):
        sm = TemporalSmoother(5, 0.65)
        self.assertEqual(sm.push(FrameObservation(0, "sitting", 0.5, True)),
                         UNCERTAIN_FRAME)
        self.assertEqual(sm.push(FrameObservation(0, "sitting", 0.8, True)),
                         "sitting")
        self.assertEqual(sm.push(FrameObservation(0, "sitting", 0.9, False)),
                         UNCERTAIN_FRAME)

    def test_sit_window_ending_while_unclear_is_uncertain(self):
        # mid-transition (low confidence) when the Sit window ends: the
        # machine cannot see whether the dog sat -> UNCERTAIN
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 2.0))  # 0.0..1.8
        v = run(s, t, "sitting", 0.5, True, 2.0)                  # unclear
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")
        self.assertAlmostEqual(s.last_t, 3.0, delta=1e-9)  # at the deadline
        self.assertIn("not clearly visible", v.reason)

    def test_sit_clearly_standing_at_deadline_is_still_early_break(self):
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 2.0))
        self.assertIsNone(run(s, t, None, 0, False, 0.2))   # t = 2.0 unclear
        v = run(s, t, "standing", 0.9, True, 2.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertIsNone(v.target_reached_at_s)

    def test_stale_smoother_after_gap_does_not_give_success(self):
        # after an unclear gap the smoother buffer is stale "sitting"; the
        # hold is only proven up to the last sitting frame (3.4 s)
        s = stay(5)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 3.6))   # 0.0..3.4
        self.assertIsNone(run(s, t, "sitting", 0.5, True, 1.4))   # unclear
        v = run(s, t, "standing", 0.9, True, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertAlmostEqual(v.break_at_s, 3.6, delta=1e-9)
        self.assertLess(v.held_s, 5.0)

    def test_standing_just_before_target_is_early_break_not_success(self):
        s = stay(5)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 4.8))   # 0.0..4.6
        v = run(s, t, "standing", 0.9, True, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertAlmostEqual(v.break_at_s, 4.8, delta=1e-9)

    def test_standing_then_leaving_view_is_early_break(self):
        s = stay(10)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 6.6))   # 0.0..6.4
        self.assertIsNone(run(s, t, "standing", 0.9, True, 0.4))  # 6.6, 6.8
        v = run(s, t, None, 0, False, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "EARLY_BREAK")
        self.assertAlmostEqual(v.break_at_s, 6.6, delta=1e-9)
        self.assertAlmostEqual(v.held_s, 6.6, delta=1e-9)

    def test_single_wrong_frame_then_lost_is_uncertain(self):
        s = stay(10)
        t = [0.0]
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 6.6))
        self.assertIsNone(run(s, t, "standing", 0.9, True, 0.2))  # 6.6 only
        v = run(s, t, None, 0, False, 3.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")

    def test_reach_confirmed_just_after_window_counts_on_raw_time(self):
        # 4 fps: lies down at raw 2.75 s, smoother confirms at 3.25 s
        s = down()
        t = [0.0]
        dt = 0.25
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 2.75, dt))
        v = run(s, t, "lying_down", 0.9, True, 3.0, dt)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertAlmostEqual(v.target_reached_at_s, 2.75, delta=1e-9)

    def test_pending_reach_does_not_defer_forever(self):
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 2.8))  # 0.0..2.6
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 0.2))   # 2.8 raw
        v = run(s, t, None, 0, False, 1.4)                        # 3.0..4.2
        self.assertIsNotNone(v)
        self.assertIsNone(v.target_reached_at_s)
        self.assertLessEqual(s.last_t, 3.0 + s.cfg.confirm_grace_s + 1e-9)

    def test_stale_reach_across_unclear_gap_is_not_backdated(self):
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 1.6))  # 0.0..1.4
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 0.2))   # 1.6 glitch
        v = run(s, t, None, 0, False, 1.4)                        # 1.8..3.0
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")
        self.assertAlmostEqual(s.last_t, 3.0, delta=1e-9)
        self.assertIsNone(v.target_reached_at_s)
        self.assertIn("not clearly visible", v.reason)
        # the late sit at 3.2 / 3.4 does not change the decided verdict
        after = run(s, t, "sitting", 0.9, True, 0.4)
        self.assertEqual(after.verdict, "UNCERTAIN")
        self.assertIsNone(after.target_reached_at_s)

    def test_real_sit_before_short_occlusion_keeps_its_reach_time(self):
        s = sit()
        t = [0.0]
        self.assertIsNone(run(s, t, "standing", 0.9, True, 2.4))  # 0.0..2.2
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 0.4))   # 2.4, 2.6
        self.assertIsNone(run(s, t, None, 0, False, 0.8))         # 2.8..3.4
        v = run(s, t, "sitting", 0.9, True, 2.0)
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertAlmostEqual(v.target_reached_at_s, 2.4, delta=1e-9)

    def test_sporadic_misreads_then_occlusion_is_uncertain(self):
        s = CommandSession("lying_down", SessionConfig(target_duration_s=10),
                           "Stay")
        t = [0.0]
        self.assertIsNone(run(s, t, "lying_down", 0.9, True, 5.2))  # ..5.0
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 0.2))     # 5.2
        self.assertIsNone(run(s, t, None, 0, False, 0.2))           # 5.4
        self.assertIsNone(run(s, t, "sitting", 0.9, True, 0.2))     # 5.6
        self.assertEqual(s.last_smoothed, "lying_down")
        v = run(s, t, None, 0, False, 3.0)                          # 5.8..
        self.assertIsNotNone(v)
        self.assertEqual(v.verdict, "UNCERTAIN")
        self.assertIsNone(v.break_at_s)
        self.assertIn("lost", v.reason)


# ---------------------------------------------------------------------------
class TestConfigAndPresets(unittest.TestCase):
    def test_defaults_match_the_app(self):
        c = SessionConfig()
        self.assertEqual((c.smooth_window, c.min_confidence,
                          c.settle_grace_s, c.break_confirm_s,
                          c.uncertain_timeout_s, c.max_uncertain_ratio,
                          c.response_window_s, c.hold_after_reach_s,
                          c.confirm_grace_s, c.reach_then_hold),
                         (5, 0.65, 2.0, 0.6, 1.5, 0.4, 3.0, 0.6, 1.0, False))

    def test_command_shape_as_in_the_app(self):
        self.assertTrue(is_reach_then_hold("Sit"))
        self.assertTrue(is_reach_then_hold("down"))
        for cmd in ("stand", "Stay", "walk", None):
            self.assertFalse(is_reach_then_hold(cmd))
        self.assertTrue(config_for_command("down", 8).reach_then_hold)
        self.assertFalse(config_for_command("stand", 8).reach_then_hold)
        self.assertFalse(config_for_command("down", 8,
                                            shape="hold").reach_then_hold)

    def test_nov11_preset_numbers(self):
        c = config_for_command("down", 8, preset="nov11")
        self.assertEqual((c.smooth_window, c.min_confidence,
                          c.break_confirm_s, c.uncertain_timeout_s,
                          c.reach_then_hold),
                         (3, 0.6, 1.0, 3.0, False))
        self.assertEqual(PRESETS["app"]["smooth_window"], 5)

    def test_frozen_nov11_module_keeps_old_reach_time(self):
        # The pre-change expectation of test_settle_within_grace, checked
        # against the frozen Nov 11 rules.
        labels = ["standing"] * 3 + ["sitting"] * 14
        obs = [state_machine_nov11.FrameObservation(t=i * FPS_DT, label=x,
                                                    confidence=0.95)
               for i, x in enumerate(labels)]
        v = state_machine_nov11.run_sequence(
            obs, "sitting", state_machine_nov11.SessionConfig(
                target_duration_s=5.0, smooth_window=3, settle_grace_s=2.0,
                break_confirm_s=1.0, uncertain_timeout_s=3.0))
        self.assertEqual(v.verdict, "SUCCESS")
        self.assertAlmostEqual(v.target_reached_at_s, 2.0)


if __name__ == "__main__":
    unittest.main()
