---
id: project2-model-c-pick-delivery
title: Project 2 Model C pick delivery
enabled: true
owner: goal:cs663-project-2-dog-training-assistant
mode: task
schedule:
  kind: runonce
  timezone: America/Los_Angeles
  at: 2026-10-25T09:12:00
delivery:
  - chat_id: 8ca9864e-fb49-451b-9712-5a6fb4cf988b
metadata:
  tags: [cron:flexible-time]
  originating_chat_context_json: '{"chat_id":"8ca9864e-fb49-451b-9712-5a6fb4cf988b","origin_provider":"main","chat_kind":"direct","event_kind":"message","require_mention":false}'
  presentation_locale: en-US
---
One-time delivery for the CS663 Project 2 group work (goal: cs663-project-2-dog-training-assistant). Send the user this message in the main chat, in your own voice, keeping it short:

For Monday's Phase 0, here is a concrete Model C pick the group can say yes or no to: MediaPipe Object Detector with EfficientDet-Lite (Lite0/Lite2) — a TFLite model trained on COCO 2017 (which includes the dog class), built for mobile CPU/GPU, with a MediaPipe Tasks Vision Android sample and a live-stream mode for camera input. Its job would be to detect the dog, draw the box, and crop/gate the frame before Model A classifies behaviour — distinct from Model A (your MobileNetV2+LSTM behaviour classifier, which still needs live camera instead of clips) and Model B (fine-tuned Gemini reasoning). No fine-tuning needed, and it uses the same LiteRT/TFLite stack as your existing APK.

Alternative if the group prefers it: YOLO11n exported to LiteRT/TFLite — also COCO-trained with dogs, but it needs more post-processing/NMS wiring into Android than MediaPipe's ready Task API. Pick it only if the group wants YOLO's newer detector or segmentation later.

One thing not to promise: dog pose. MediaPipe Pose is human pose; a true dog-pose model would be a research integration, not a Phase 0 pick. If the scope is sit/stay/come posture, the detector still works as stage 1 and posture stays with Model A, with Gemini (B) narrating/verifying.

One question for him: take EfficientDet-Lite via MediaPipe as Model C, or YOLO11n instead, with Model C's job defined as dog detection + crop/gating (not behaviour classification)? This is a proposal for the group, not a settled decision — the roster and app scope are not locked yet.

Do not message the group or anyone else; do not submit anything. This is a one-shot delivery — after sending, the job is done.
