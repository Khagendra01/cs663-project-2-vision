# Base-model evaluation results — CS663 Gemini exercise

Measured Oct 30, 2026, ~08:27–08:39 AM PT, Cloud Shell, project cs663-dog-training,
google-genai SDK, location 'global', exact training prompt, strict exact-match scoring
(lowercase/strip; answer must be exactly one of sitting/standing/walking/lying down).

Test set: 43 held-out images (sitting 9, standing 11, walking 9, lying down 14).
Per-image predictions saved in the account's Cloud Shell home:
~/base_eval_gemini-3.5-flash_global.json and ~/base_eval_gemini-3-flash-preview_global.json

## gemini-3.5-flash (Run 1's base) — 37/43 = 86.0%, 3 invalid, 0 API errors
Per-class: sitting 8/9 = 88.9%; standing 8/11 = 72.7%; walking 8/9 = 88.9%; lying down 13/14 = 92.9%.
Confusion (rows true, cols predicted; sitting/standing/walking/lying down):
- sitting: 8, 1, 0, 0
- standing: 0, 8, 1, 0 (+2 invalid)
- walking: 0, 1, 8, 0
- lying down: 0, 0, 0, 13 (+1 invalid)
All 3 invalids were format violations with the correct label inside a sentence
("The dog is standing.", "The dog is **standing**.", "The dog is **lying down**.").
Lenient label extraction would give 40/43 = 93.0%.

## gemini-3-flash-preview (reference base) — 40/43 = 93.0%, 1 invalid, 0 errors after one 429 retry
Per-class: sitting 9/9 = 100%; standing 10/11 = 90.9%; walking 7/9 = 77.8%; lying down 14/14 = 100%.
Confusion:
- sitting: 9, 0, 0, 0
- standing: 0, 10, 1, 0
- walking: 0, 1, 7, 0 (+1 invalid: "The dog is walking.")
- lying down: 0, 0, 0, 14
Note: first pass was 39/43 with index 25 (lying down) failing on a 429 RESOURCE_EXHAUSTED;
single retry returned 'lying down' (correct); saved file reflects 40/43.
Lenient label extraction would give 41/43 = 95.3%.

## gemini-2.5-flash (Run 2's base) — NOT callable
404 "Publisher model ... was not found or your project does not have access to it"
at BOTH us-central1 and global, every SDK. No base number exists for it; do not estimate one.
If Run 2 wins, its tuned model must be tested via its own endpoint resource name,
and the base comparison is disclosed as unavailable for 2.5 (3.5-flash base used as reference).
