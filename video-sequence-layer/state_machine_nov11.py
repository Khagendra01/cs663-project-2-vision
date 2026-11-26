#!/usr/bin/env python3
"""FROZEN Nov 11 2026 version of the temporal state machine.

Kept unchanged (apart from this paragraph) so the Nov 10-11 results stay
exactly reproducible: results/verification_run.txt, the Nov 11 sections of
results/video_evaluation.md, scenarios.json and
../gemini-finetune/model-b-scenarios/scenarios.jsonl were all judged by
this code (commit 947da56). The current rules - the same as the Android
app - are in state_machine.py; use that file for anything new.

Original description follows.

Temporal state machine for CS663 Project 2 - Dog Training Assistant.

This is the video layer Grewe's proposal feedback asked for: the app does
NOT judge a command from one still image. A command interval starts
explicitly (Start Sequence in the app), a frame-level posture classifier
(Model A on device, or a Gemini endpoint per selected frame) emits one
label per sampled frame, and this machine turns that *sequence* into one
verdict:

    SUCCESS      the dog reached the target posture and held it for the
                 target duration
    EARLY_BREAK  the dog held the target, then left it (sustained, after
                 smoothing) before the target duration was reached
    UNCERTAIN    the evidence is too weak to judge: target never
                 established, too many low-confidence / no-dog frames, the
                 dog was lost for too long, or the interval ended before
                 the target duration

Temporal smoothing: a majority vote over a sliding window of raw labels.
Single-frame noise (one wrong label in a clean run) never flips a
verdict; a break must be sustained for `break_confirm_s` of smoothed
non-target labels. Frames below `min_confidence`, or frames with no dog,
become the special label UNCERTAIN_FRAME and neither count for nor
against the dog - but too many of them make the whole attempt UNCERTAIN.

Labels are canonicalised: sit -> sitting, down / lie -> lying_down,
stand -> standing, walk -> walking. Commands map to targets with
COMMAND_TARGETS; "stay" takes whatever posture it is paired with, so the
session always takes an explicit target_label.

Pure stdlib, no I/O: the Android app mirrors this logic, and
sequence_evaluate.py drives it from video files / label files.
"""
from collections import deque
from dataclasses import dataclass, field

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


def canonical_label(raw):
    """Map a classifier's raw text to a canonical label or UNCERTAIN_FRAME."""
    if raw is None:
        return UNCERTAIN_FRAME
    return _ALIASES.get(str(raw).strip().lower(), UNCERTAIN_FRAME)


@dataclass
class FrameObservation:
    t: float                      # seconds since Start Sequence
    label: str | None             # raw classifier label (any alias/case)
    confidence: float | None = None
    dog_present: bool = True


@dataclass
class SessionConfig:
    target_duration_s: float = 5.0
    min_confidence: float = 0.6       # below this a frame is uncertain
    smooth_window: int = 5            # majority-vote window (frames)
    settle_grace_s: float = 2.0       # time allowed to first reach target
    break_confirm_s: float = 1.0      # sustained smoothed break required
    uncertain_timeout_s: float = 3.0  # lost/unclear for this long -> UNCERTAIN
    max_uncertain_ratio: float = 0.4  # overall unclear share -> UNCERTAIN


@dataclass
class Verdict:
    verdict: str                      # SUCCESS | EARLY_BREAK | UNCERTAIN
    reason: str
    target_label: str
    target_duration_s: float
    target_reached_at_s: float | None = None
    held_s: float = 0.0               # time target was held before break/end
    # when the break began: the FIRST raw observation of the wrong
    # posture, not the later confirmation frame (updated proposal 5.3:
    # "record the break timestamp at the first observation of that
    # confirmed sustained change"). The smoother lags the raw stream by
    # up to smooth_window - 1 frames, so break_at_s is earlier than the
    # frame where the break was confirmed.
    break_at_s: float | None = None
    n_frames: int = 0
    uncertain_ratio: float = 0.0
    smoothed_sequence: list = field(default_factory=list)  # [(t, label)]


class TemporalSmoother:
    """Majority vote over the last `window` usable (confident, dog) labels.

    Uncertain frames pass through as UNCERTAIN_FRAME and do not enter the
    window, so one blurry frame cannot vote a good run down.
    """

    def __init__(self, window=5, min_confidence=0.6):
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
        best = max(reversed(self._buf), key=lambda x: (counts[x],))
        return best


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
        # t of the first RAW (unsmoothed) wrong-posture observation
        # while the hold was in progress; becomes break_at_s if the
        # break confirms (proposal 5.3). Reset whenever the smoothed
        # label returns to the target, so a one-frame glitch is forgotten.
        self._pending_break_start: float | None = None
        self._uncertain_since: float | None = None
        self._last_t: float | None = None
        self._n = 0
        self._n_uncertain = 0
        self._seq: list = []

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
        self._n += 1
        self._last_t = obs.t
        # The raw label, if it is a usable confident dog observation at
        # all - this is what the break timestamp is taken from (proposal
        # 5.3), before smoothing delays it by up to smooth_window - 1
        # frames.
        raw = canonical_label(obs.label)
        if (not obs.dog_present or raw == UNCERTAIN_FRAME
                or (obs.confidence is not None
                    and obs.confidence < self.cfg.min_confidence)):
            raw = UNCERTAIN_FRAME
        smoothed = self.smoother.push(obs)
        if smoothed == UNCERTAIN_FRAME:
            self._n_uncertain += 1
        self._seq.append((round(obs.t, 3), smoothed))

        # First raw sighting of a wrong posture during the hold. (Before
        # the target is reached the dog is simply not there yet, so
        # nothing is recorded.)
        if (self._reached_at is not None and raw != UNCERTAIN_FRAME
                and raw != self.target
                and self._pending_break_start is None):
            self._pending_break_start = obs.t

        # Lost / unclear for too long -> UNCERTAIN (only while tracking;
        # before the target is reached the settle grace governs instead).
        if smoothed == UNCERTAIN_FRAME:
            if self._uncertain_since is None:
                self._uncertain_since = obs.t
            if (self._reached_at is not None
                    and obs.t - self._uncertain_since
                    >= self.cfg.uncertain_timeout_s):
                return self._make(
                    "UNCERTAIN",
                    f"dog lost / unclear for {self.cfg.uncertain_timeout_s:.0f}s "
                    "during the hold")
        else:
            self._uncertain_since = None

        if self._reached_at is None:
            if smoothed == self.target:
                self._reached_at = obs.t
            elif obs.t >= self.cfg.settle_grace_s:
                return self._make(
                    "UNCERTAIN",
                    f"target posture '{self.target}' never established "
                    f"within the {self.cfg.settle_grace_s:.0f}s settle grace")
            return None

        # Tracking the hold.
        if smoothed == self.target:
            self._break_since = None
            # A wrong raw frame the smoother voted down is a glitch:
            # forget it only once a RAW frame shows the dog back in the
            # target posture. Do NOT reset on a smoothed target while
            # the raw stream still shows the wrong posture - the
            # smoother lags the raw stream by up to smooth_window - 1
            # frames, and resetting here would erase the break's true
            # start time (this bug was caught by the scen02 check: the
            # break reported 3.0s instead of the raw 2.5s).
            if raw == self.target:
                self._pending_break_start = None
            if obs.t - self._reached_at >= self.cfg.target_duration_s:
                return self._make(
                    "SUCCESS",
                    f"held '{self.target}' for the full "
                    f"{self.cfg.target_duration_s:.0f}s target duration")
            return None

        if smoothed != UNCERTAIN_FRAME:
            if self._break_since is None:
                self._break_since = obs.t
            if obs.t - self._break_since >= self.cfg.break_confirm_s:
                # Break timestamp = first raw sighting of the wrong
                # posture (proposal 5.3), not the confirmation frame.
                # Fall back to the smoothed start if no raw sighting
                # was recorded (cannot happen in practice: a smoothed
                # non-target label implies a raw one preceded it).
                break_at = (self._pending_break_start
                            if self._pending_break_start is not None
                            else self._break_since)
                return self._make(
                    "EARLY_BREAK",
                    f"left '{self.target}' after holding "
                    f"{break_at - self._reached_at:.1f}s of the "
                    f"{self.cfg.target_duration_s:.0f}s target",
                    break_at=break_at,
                    held=break_at - self._reached_at)
        return None

    def finish(self) -> Verdict:
        """Verdict for an interval that ended with no decision yet."""
        if self._verdict is not None:
            return self._verdict
        if self._n and (self._n_uncertain / self._n
                        > self.cfg.max_uncertain_ratio):
            return self._make(
                "UNCERTAIN",
                f"{self._n_uncertain}/{self._n} frames were low-confidence "
                "or had no dog - too unclear to judge")
        if self._reached_at is None:
            return self._make(
                "UNCERTAIN",
                f"target posture '{self.target}' was never established")
        return self._make(
            "UNCERTAIN",
            f"interval ended after holding "
            f"{self._last_t - self._reached_at:.1f}s of the "
            f"{self.cfg.target_duration_s:.0f}s target duration - "
            "incomplete, not a success and not a break")


def run_sequence(observations, target_label, config=None, command=None):
    """Convenience: feed a whole list and return the final Verdict."""
    s = CommandSession(target_label, config, command)
    for obs in observations:
        v = s.feed(obs)
        if v is not None:
            return v
    return s.finish()
