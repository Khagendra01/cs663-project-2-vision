#!/usr/bin/env python3
"""Temporal state machine for CS663 Project 2 - Dog Training Assistant.

This is the video layer Grewe's proposal feedback asked for: the app does
NOT judge a command from one still image. A command interval starts
explicitly (Start Sequence in the app), a frame-level posture classifier
(Model A on device, or a Gemini endpoint per selected frame) emits one
label per sampled frame, and this machine turns that *sequence* into one
verdict:

    SUCCESS      the dog reached the target posture and held it for the
                 required time
    EARLY_BREAK  the dog held the target, then left it (sustained, after
                 smoothing) before the required time - or, for Sit / Down,
                 was clearly seen NOT reaching it within the response
                 window (a failed command, held 0 s)
    UNCERTAIN    the evidence is too weak to judge: target never
                 established, too many low-confidence / no-dog frames, the
                 dog was lost for too long, the dog was not clearly
                 visible at the Sit / Down deadline, or the interval ended
                 before the required time

Nov 25 2026 revision: same rules as the Android app
-------------------------------------------------
The Android app's Java port (cs663-project-2-app, SequenceSession.java,
21 JUnit tests) gained rules after this file's Nov 11 version. This
revision brings the Python reference back in line with it: same rules,
same defaults, same reason strings. The app's unit tests are ported to
test_state_machine.py.

Time base: t is real elapsed time in seconds since Start Sequence, not a
frame index. The app classifies frames on the device and is designed for
about 5 fps (its only on-device timing so far comes from the Android
emulator; no physical phone has been used yet). Stored 2 fps frames are
fed at t = frame_index / 2. Every time below is in seconds:

    smooth_window       5     majority vote over the last 5 usable frames
    min_confidence      0.65  softmax below this -> unclear frame
    settle_grace_s      2.0   HOLD commands: target must appear this fast
    break_confirm_s     0.6   a smoothed non-target run this long is a break
    uncertain_timeout_s 1.5   unclear / no dog this long -> UNCERTAIN
                              (the app calls it lostS)
    max_uncertain_ratio 0.4   checked in finish()
    response_window_s   3.0   Sit / Down: reach the target within 3 s ...
    hold_after_reach_s  0.6   ... then hold it 0.6 s
    confirm_grace_s     1.0   the smoother may confirm an in-window reach
                              up to this long after the deadline

Two command shapes (use config_for_command() to pick one the way the
app does):
  - HOLD (Stay / Stand / Walk): reach the target within the settle grace
    and hold it for the full target duration.
  - REACH-THEN-HOLD (Sit / Down, reach_then_hold=True): reach the target
    within the response window, then hold it hold_after_reach_s. Window
    expired, dog clearly seen in another posture -> EARLY_BREAK held 0 s.
    Dog unclear at the deadline (or within break_confirm_s of it) ->
    UNCERTAIN ("dog not clearly visible"), since it may have responded
    out of sight.

Both ends of the hold are measured on the RAW frame stream, because the
smoother lags it by a few frames:
  - reach time = first raw target frame of the run the smoother then
    confirmed. A run that started inside the window but is confirmed up
    to confirm_grace_s after it still counts as in time. A LONE pending
    raw target frame does not survive an unclear gap of break_confirm_s
    or longer (it says nothing about when the dog really got there); two
    or more confident target frames before a short occlusion do.
  - break time = first frame after the LAST raw target frame: the first
    wrong raw frame, or the start of the unclear gap before it. A single
    wrong raw frame the smoother votes down is forgotten as soon as a
    raw frame shows the target again.
  - SUCCESS needs the smoothed AND the raw label to be the target on the
    deciding frame, so a stale smoother buffer (lag, or an unclear gap
    that never flushed it) cannot declare a hold the dog already broke.
  - Lost / unclear for uncertain_timeout_s, in any phase -> UNCERTAIN,
    unless the wrong posture was already seen after the reach (a smoothed
    break run, or >= 2 CONSECUTIVE wrong raw frames as the last usable
    frames before the gap): then the dog left the posture and then the
    view -> EARLY_BREAK at the first wrong frame. Sporadic misreads with
    unclear frames between them are not evidence of a break.

Nov 11 settings: the end-to-end video evaluation and the scenario files
of Nov 10-11 were produced with smooth_window 3, min_confidence 0.6,
break_confirm_s 1.0, uncertain_timeout_s 3.0, every command judged as a
HOLD, and t = frame index. PRESETS["nov11"] holds those numbers.
The Nov 11 *rules* (smoothed reach time, lost rule only after the reach,
no response window) are kept verbatim in state_machine_nov11.py, so the
old results can be reproduced exactly; see results/video_evaluation.md.

Labels are canonicalised: sit -> sitting, down / lie -> lying_down,
stand -> standing, walk -> walking. Commands map to targets with
COMMAND_TARGETS; "stay" takes whatever posture it is paired with, so the
session always takes an explicit target_label.

Pure stdlib, no I/O: sequence_evaluate.py drives it from video files /
label files, and the Android app's SequenceSession.java mirrors it.
"""
from collections import deque
from dataclasses import dataclass, field, fields
from decimal import ROUND_HALF_UP, Decimal

LABELS = ("sitting", "standing", "lying_down", "walking")
UNCERTAIN_FRAME = "uncertain_frame"

_ALIASES = {
    "sit": "sitting", "sitting": "sitting",
    "stand": "standing", "standing": "standing",
    "down": "lying_down", "lie": "lying_down", "lying": "lying_down",
    "lying down": "lying_down", "lying_down": "lying_down",
    "walk": "walking", "walking": "walking",
    "transition": UNCERTAIN_FRAME, "undefined": UNCERTAIN_FRAME,
    "none": UNCERTAIN_FRAME, "no_dog": UNCERTAIN_FRAME, "": UNCERTAIN_FRAME,
}

COMMAND_TARGETS = {
    "sit": "sitting",
    "down": "lying_down",
    "stand": "standing",
    "walk": "walking",
    "heel": "walking",
    # "stay" / "stay in <posture>" must pass target_label explicitly.
}

# Commands judged as REACH-THEN-HOLD, as in the app
# (SequenceActivity.isReachThenHold: Sit and Down).
REACH_THEN_HOLD_COMMANDS = ("sit", "down")


def canonical_label(raw):
    """Map a classifier's raw text to a canonical label or UNCERTAIN_FRAME."""
    if raw is None:
        return UNCERTAIN_FRAME
    return _ALIASES.get(str(raw).strip().lower(), UNCERTAIN_FRAME)


def is_reach_then_hold(command):
    """True for the commands the app judges as reach-then-hold."""
    return (command is not None
            and str(command).strip().lower() in REACH_THEN_HOLD_COMMANDS)


def _fmt(x, places):
    """%.Nf with half-up rounding, as Java's String.format does it."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def _fmt0(x):
    return _fmt(x, 0)


def _fmt1(x):
    return _fmt(x, 1)


@dataclass
class FrameObservation:
    t: float                      # seconds since Start Sequence
    label: str | None             # raw classifier label (any alias/case)
    confidence: float | None = None
    dog_present: bool = True


@dataclass
class SessionConfig:
    target_duration_s: float = 5.0
    min_confidence: float = 0.65      # below this a frame is unclear
    smooth_window: int = 5            # majority-vote window (frames)
    settle_grace_s: float = 2.0       # HOLD: time allowed to first reach
    break_confirm_s: float = 0.6      # sustained smoothed break required
    uncertain_timeout_s: float = 1.5  # lost/unclear this long -> UNCERTAIN
    max_uncertain_ratio: float = 0.4  # overall unclear share -> UNCERTAIN
    # Sit / Down shape (proposal 5.2): reach the target within
    # response_window_s, then hold it for hold_after_reach_s.
    reach_then_hold: bool = False
    response_window_s: float = 3.0
    hold_after_reach_s: float = 0.6
    # A raw target run that began inside the response window (or the
    # settle grace) may be confirmed by the smoother up to this long
    # after the deadline.
    confirm_grace_s: float = 1.0


_CONFIG_FIELDS = {f.name for f in fields(SessionConfig)}

# Named parameter sets. "app" = the Android app's values (the defaults
# above). "nov11" = the numbers the Nov 11 video evaluation and the
# scenario files used. NOTE: "nov11" only restores the NUMBERS; the
# Nov 11 rules themselves are in state_machine_nov11.py.
PRESETS = {
    "app": dict(min_confidence=0.65, smooth_window=5, settle_grace_s=2.0,
                break_confirm_s=0.6, uncertain_timeout_s=1.5,
                max_uncertain_ratio=0.4),
    "nov11": dict(min_confidence=0.6, smooth_window=3, settle_grace_s=2.0,
                  break_confirm_s=1.0, uncertain_timeout_s=3.0,
                  max_uncertain_ratio=0.4),
}


def config_for_command(command, target_duration_s=5.0, preset="app",
                       shape="auto", **overrides):
    """SessionConfig for a command, chosen the way the app chooses it.

    shape: "auto" (Sit / Down -> reach-then-hold, everything else ->
    hold, as in the app), "hold" or "reach". The "nov11" preset judged
    every command as a hold, so with preset="nov11" "auto" means "hold".
    Keyword overrides are applied last (unknown names raise TypeError).
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; one of {list(PRESETS)}")
    if shape not in ("auto", "hold", "reach"):
        raise ValueError(f"unknown shape {shape!r}")
    kw = dict(PRESETS[preset])
    if shape == "auto":
        kw["reach_then_hold"] = (preset == "app"
                                 and is_reach_then_hold(command))
    else:
        kw["reach_then_hold"] = shape == "reach"
    bad = set(overrides) - _CONFIG_FIELDS
    if bad:
        raise TypeError(f"unknown SessionConfig field(s): {sorted(bad)}")
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return SessionConfig(target_duration_s=float(target_duration_s), **kw)


@dataclass
class Verdict:
    verdict: str                      # SUCCESS | EARLY_BREAK | UNCERTAIN
    reason: str
    target_label: str
    target_duration_s: float
    target_reached_at_s: float | None = None
    held_s: float = 0.0               # time target was held before break/end
    # when the break began: the first frame after the last raw target
    # frame (updated proposal 5.3: "record the break timestamp at the
    # first observation of that confirmed sustained change"), not the
    # later confirmation frame.
    break_at_s: float | None = None
    n_frames: int = 0
    uncertain_ratio: float = 0.0
    smoothed_sequence: list = field(default_factory=list)  # [(t, label)]


class TemporalSmoother:
    """Majority vote over the last `window` usable (confident, dog) labels.

    Uncertain frames pass through as UNCERTAIN_FRAME and do not enter the
    window, so one blurry frame cannot vote a good run down. Ties go to
    the most recent of the tied labels.
    """

    def __init__(self, window=5, min_confidence=0.65):
        self.window = max(1, window)
        self.min_confidence = min_confidence
        self._buf = deque(maxlen=self.window)

    def push(self, obs: FrameObservation) -> str:
        label = canonical_label(obs.label)
        if (not obs.dog_present or label == UNCERTAIN_FRAME
                or (obs.confidence is not None
                    and obs.confidence < self.min_confidence)):
            return UNCERTAIN_FRAME
        self._buf.append(label)
        counts = {}
        for x in self._buf:
            counts[x] = counts.get(x, 0) + 1
        # most common; ties go to the most recent among the tied
        return max(reversed(self._buf), key=lambda x: counts[x])


class CommandSession:
    """One command interval, from Start Sequence to verdict.

    Feed FrameObservations in time order with feed(); call finish() when
    the interval/stream ends if no verdict was reached during feeding.
    feed() returns a Verdict as soon as one is decided, else None.
    """

    def __init__(self, target_label, config: SessionConfig | None = None,
                 command: str | None = None):
        self.target = canonical_label(target_label)
        if self.target == UNCERTAIN_FRAME:
            raise ValueError(f"unknown target label: {target_label!r}")
        self.cfg = config or SessionConfig()
        self.command = command
        self.smoother = TemporalSmoother(self.cfg.smooth_window,
                                         self.cfg.min_confidence)
        self._verdict: Verdict | None = None
        self._reached_at: float | None = None
        self._break_since: float | None = None
        # Start of the break on the raw time base (see the module
        # docstring); becomes break_at_s if the break confirms. Reset only
        # when a raw frame shows the dog back in the target posture -
        # never on a smoothed target alone, because the smoother lags the
        # raw stream and would erase the break's true start time.
        self._pending_break_start: float | None = None
        # Length of the current run of CONSECUTIVE usable wrong raw frames
        # while holding. A raw target frame resets it; an unclear frame
        # ends the run, so the next wrong frame starts a new run at 1.
        self._wrong_raw_streak = 0
        self._prev_raw_unclear = False
        # First raw target frame of the current run while the target is
        # not yet reached (becomes the reach time when the smoother
        # confirms), and how many raw target frames that run has.
        # Unclear frames leave it alone; a usable non-target frame resets.
        self._pending_reach_start: float | None = None
        self._pending_reach_frames = 0
        # t of the latest raw target frame, and of the first frame (any
        # kind) after it - where a break starts.
        self._last_raw_target_at: float | None = None
        self._first_after_raw_target: float | None = None
        self._last_unclear_t: float | None = None
        self._uncertain_since: float | None = None
        self._last_t: float | None = None
        self._last_smoothed: str | None = None
        self._n = 0
        self._n_uncertain = 0
        self._seq: list = []

    # -- read-only state, for live UIs and tests -------------------------
    @property
    def last_t(self):
        return self._last_t

    @property
    def last_smoothed(self):
        return self._last_smoothed

    @property
    def reached_at(self):
        return self._reached_at

    @property
    def frame_count(self):
        return self._n

    @property
    def is_decided(self):
        return self._verdict is not None

    @property
    def phase(self):
        """RESPONSE (Sit/Down, not reached yet), HOLD, or DONE."""
        if self._verdict is not None:
            return "DONE"
        if not self.cfg.reach_then_hold:
            return "HOLD"
        return "RESPONSE" if self._reached_at is None else "HOLD"

    def hold_required_s(self):
        """Full target duration (HOLD) or the short hold (Sit / Down)."""
        return (self.cfg.hold_after_reach_s if self.cfg.reach_then_hold
                else self.cfg.target_duration_s)

    # -- internals -----------------------------------------------------
    def _make(self, verdict, reason, break_at=None, held=None):
        self._verdict = Verdict(
            verdict=verdict, reason=reason, target_label=self.target,
            target_duration_s=self.cfg.target_duration_s,
            target_reached_at_s=self._reached_at,
            held_s=(self._last_t - self._reached_at
                    if held is None and self._reached_at is not None
                    and self._last_t is not None else (held or 0.0)),
            break_at_s=break_at, n_frames=self._n,
            uncertain_ratio=(self._n_uncertain / self._n if self._n else 0.0),
            smoothed_sequence=list(self._seq))
        return self._verdict

    # -- main API --------------------------------------------------------
    def feed(self, obs: FrameObservation) -> Verdict | None:
        if self._verdict is not None:
            return self._verdict
        cfg = self.cfg
        tgt = self.target
        t = obs.t
        self._n += 1
        self._last_t = t
        # The raw label, if it is a usable observation at all - the reach
        # and break timestamps are taken from it, before smoothing delays it.
        raw = canonical_label(obs.label)
        if (not obs.dog_present or raw == UNCERTAIN_FRAME
                or (obs.confidence is not None
                    and obs.confidence < cfg.min_confidence)):
            raw = UNCERTAIN_FRAME
        smoothed = self.smoother.push(obs)
        self._last_smoothed = smoothed
        if smoothed == UNCERTAIN_FRAME:
            self._n_uncertain += 1
        self._seq.append((round(t, 3), smoothed))

        # Raw-time bookkeeping: both ends of the hold are measured on raw
        # frames, not on the lagged smoothed stream.
        raw_unclear = raw == UNCERTAIN_FRAME
        if raw_unclear:
            self._last_unclear_t = t
        if raw == tgt:
            self._last_raw_target_at = t
            self._first_after_raw_target = None
        elif (self._last_raw_target_at is not None
              and self._first_after_raw_target is None):
            self._first_after_raw_target = t
        if self._reached_at is None:
            if raw == tgt:
                if self._pending_reach_start is None:
                    self._pending_reach_start = t
                    self._pending_reach_frames = 0
                self._pending_reach_frames += 1
            elif not raw_unclear:
                self._pending_reach_start = None
                self._pending_reach_frames = 0
        after_unclear = self._prev_raw_unclear
        self._prev_raw_unclear = raw_unclear
        if raw == tgt:
            self._wrong_raw_streak = 0

        # Lost / unclear for too long -> UNCERTAIN, in any phase: an
        # attempt that cannot be seen cannot be judged.
        if smoothed == UNCERTAIN_FRAME:
            if self._uncertain_since is None:
                self._uncertain_since = t
            if t - self._uncertain_since >= cfg.uncertain_timeout_s:
                # The dog left the target posture and THEN left the view.
                # That is a broken hold, but only with real evidence: a
                # smoothed wrong posture, or >= 2 consecutive wrong raw
                # frames.
                if self._reached_at is not None and (
                        self._break_since is not None
                        or (self._pending_break_start is not None
                            and self._wrong_raw_streak >= 2)):
                    break_at = (self._pending_break_start
                                if self._pending_break_start is not None
                                else self._break_since)
                    return self._make(
                        "EARLY_BREAK",
                        f"left '{tgt}' after holding "
                        f"{_fmt1(break_at - self._reached_at)}s, "
                        "then left the view",
                        break_at=break_at,
                        held=break_at - self._reached_at)
                return self._make(
                    "UNCERTAIN",
                    f"dog lost / unclear for "
                    f"{_fmt1(t - self._uncertain_since)}s in a row - "
                    "cannot judge the attempt")
        else:
            self._uncertain_since = None

        # A lone pending reach frame must not survive an unclear run of
        # break_confirm_s or longer: after a gap that long it says nothing
        # about when the dog actually got there. Two or more confident
        # target frames before a short occlusion are a real reach and keep
        # their start time. The 1e-6 lets an exactly-break_confirm_s run
        # count despite float rounding.
        if (self._reached_at is None and raw_unclear
                and self._pending_reach_start is not None
                and self._pending_reach_frames < 2
                and self._uncertain_since is not None
                and t - self._uncertain_since
                >= cfg.break_confirm_s - 1e-6):
            self._pending_reach_start = None
            self._pending_reach_frames = 0

        # First raw sighting of a wrong posture during the hold, stamped
        # at the first frame after the last raw target frame (with no gap
        # that is this frame; after an unclear gap, the gap's start).
        if self._reached_at is not None and not raw_unclear and raw != tgt:
            if self._pending_break_start is None:
                fa = self._first_after_raw_target
                self._pending_break_start = (
                    fa if fa is not None and fa >= self._reached_at else t)
            self._wrong_raw_streak = (1 if after_unclear
                                      else self._wrong_raw_streak + 1)

        if self._reached_at is None:
            deadline = (cfg.response_window_s if cfg.reach_then_hold
                        else cfg.settle_grace_s)
            if smoothed == tgt:
                # Reach time on the raw time base.
                self._reached_at = (self._pending_reach_start
                                    if self._pending_reach_start is not None
                                    else t)
                self._pending_reach_start = None
                self._pending_reach_frames = 0
                if (cfg.reach_then_hold
                        and self._reached_at > cfg.response_window_s):
                    return self._make(
                        "EARLY_BREAK",
                        f"target posture '{tgt}' reached at "
                        f"{_fmt1(self._reached_at)}s, after the "
                        f"{_fmt0(cfg.response_window_s)}s response window",
                        held=0.0)
                # Fall through: the hold may already be complete here.
            else:
                if t < deadline:
                    return None
                # A raw target run that began inside the window is still
                # being confirmed by the smoother: wait, but only up to
                # confirm_grace_s past the deadline.
                if (self._pending_reach_start is not None
                        and self._pending_reach_start <= deadline
                        and t < deadline + cfg.confirm_grace_s):
                    return None
                if not cfg.reach_then_hold:
                    return self._make(
                        "UNCERTAIN",
                        f"target posture '{tgt}' never established within "
                        f"the {_fmt0(cfg.settle_grace_s)}s settle grace")
                # RESPONSE phase (Sit / Down). Unclear now, or within
                # break_confirm_s: the dog may be mid-transition out of
                # sight, so the command cannot be judged.
                recently_unclear = (
                    smoothed == UNCERTAIN_FRAME
                    or (self._last_unclear_t is not None
                        and t - self._last_unclear_t < cfg.break_confirm_s))
                if recently_unclear:
                    return self._make(
                        "UNCERTAIN",
                        "dog not clearly visible at the end of the "
                        f"{_fmt0(cfg.response_window_s)}s response window "
                        "- cannot judge the response")
                # Clearly seen in another posture to the end: a failed
                # command, hence EARLY_BREAK with held 0 s.
                return self._make(
                    "EARLY_BREAK",
                    f"target posture '{tgt}' not reached within the "
                    f"{_fmt0(cfg.response_window_s)}s response window",
                    held=0.0)

        # Tracking the hold.
        if smoothed == tgt:
            self._break_since = None
            # A wrong raw frame the smoother voted down is a glitch:
            # forget it only once a RAW frame shows the target again.
            if raw == tgt:
                self._pending_break_start = None
            # The deciding frame must itself show the target (raw).
            if raw == tgt and t - self._reached_at >= self.hold_required_s():
                if cfg.reach_then_hold:
                    return self._make(
                        "SUCCESS",
                        f"reached '{tgt}' within the "
                        f"{_fmt0(cfg.response_window_s)}s response window "
                        f"and held it for the "
                        f"{_fmt1(cfg.hold_after_reach_s)}s confirmation hold")
                return self._make(
                    "SUCCESS",
                    f"held '{tgt}' for the full "
                    f"{_fmt0(cfg.target_duration_s)}s target duration")
            return None

        if smoothed != UNCERTAIN_FRAME:
            if self._break_since is None:
                self._break_since = t
            if t - self._break_since >= cfg.break_confirm_s:
                # Break timestamp = first raw sighting of the wrong
                # posture (proposal 5.3), not the confirmation frame.
                break_at = (self._pending_break_start
                            if self._pending_break_start is not None
                            else self._break_since)
                held = break_at - self._reached_at
                if cfg.reach_then_hold:
                    return self._make(
                        "EARLY_BREAK",
                        f"reached '{tgt}' but left it after holding "
                        f"{_fmt1(held)}s of the "
                        f"{_fmt1(cfg.hold_after_reach_s)}s confirmation hold",
                        break_at=break_at, held=held)
                return self._make(
                    "EARLY_BREAK",
                    f"left '{tgt}' after holding {_fmt1(held)}s of the "
                    f"{_fmt0(cfg.target_duration_s)}s target",
                    break_at=break_at, held=held)
        return None

    def finish(self) -> Verdict:
        """Verdict for an interval that ended with no decision yet."""
        if self._verdict is not None:
            return self._verdict
        cfg = self.cfg
        if self._n and (self._n_uncertain / self._n
                        > cfg.max_uncertain_ratio):
            return self._make(
                "UNCERTAIN",
                f"{self._n_uncertain}/{self._n} frames had no usable label "
                "(no dog detected or low model confidence) - too unclear "
                "to judge")
        if self._reached_at is None:
            return self._make(
                "UNCERTAIN",
                f"target posture '{self.target}' was never established")
        held = _fmt1(self._last_t - self._reached_at)
        if cfg.reach_then_hold:
            return self._make(
                "UNCERTAIN",
                f"interval ended after holding {held}s of the "
                f"{_fmt1(cfg.hold_after_reach_s)}s confirmation hold - "
                "incomplete, not a success and not a break")
        return self._make(
            "UNCERTAIN",
            f"interval ended after holding {held}s of the "
            f"{_fmt0(cfg.target_duration_s)}s target duration - "
            "incomplete, not a success and not a break")


def run_sequence(observations, target_label, config=None, command=None):
    """Convenience: feed a whole list and return the final Verdict."""
    s = CommandSession(target_label, config, command)
    for obs in observations:
        v = s.feed(obs)
        if v is not None:
            return v
    return s.finish()
