# Tuned-model evaluation results, CS663 Gemini exercise

Measured Oct 30, 2026, about 6:26 PM PT, Cloud Shell, project cs663-dog-training.
Per-image predictions saved in the account's Cloud Shell home: ~/tuned_eval_winner.json

## Winning job
Job A SUCCEEDED: tuning job 1708100736908263424, base gemini-3.5-flash, us-central1.
Created Oct 29, 11:56 PM PT; finished Oct 30, 4:53 PM PT (about 16 h 57 m, the known stall).
Tuned model: projects/917815040923/locations/us/models/83175855617998848@1
IMPORTANT serving detail: the tuned model does NOT answer when called by its model
resource name (404 in us-central1, global, and us). It serves through its deployed
endpoint with client location 'us':
Endpoint: projects/917815040923/locations/us/endpoints/514008491845746688
For the Nov 6 lab backend: TUNED_ENDPOINT should be that endpoint resource name and
VERTEX_LOCATION should be 'us' for tuned calls.

## Tuned accuracy (43 held out test images, strict exact match, same prompt as training)
Overall: 38/43 = 88.4%. Invalid: 3. API errors: 0.
Base gemini-3.5-flash for comparison: 37/43 = 86.0%. Tuned is +1 image net.
Lenient scoring (right label inside a sentence counted correct): tuned 40/43, base 40/43.

Per class (tuned): sitting 8/9 = 88.9%; standing 8/11 = 72.7%; walking 8/9 = 88.9%; lying down 14/14 = 100%.
Per class (base): sitting 8/9; standing 8/11; walking 8/9; lying down 13/14. The clear gain is lying down.

Confusion matrix (tuned; rows true, cols predicted; sitting/standing/walking/lying down):
sitting: 8, 0, 0, 0 (plus 1 invalid)
standing: 0, 8, 1, 0 (plus 2 invalid)
walking: 0, 1, 8, 0
lying down: 0, 0, 0, 14

All 5 misses: 2 wrong labels (standing photo called walking; walking photo called standing)
and 3 verbose answers. 2 of the 3 verbose answers had the correct label inside the sentence,
so fine tuning improved accuracy a little but did not fully fix the format problem at 1 epoch.

Spot checks (tuned):
- ph_acabashi_standing_001.jpg, true standing, output "walking" (wrong; same image the base missed)
- ph_f_standing_002.jpg, true standing, output "Based on the image, the dog is **standing**." (invalid format)
- ph_gazebo_sitting_001.jpg, true sitting, output "The dog is sitting." (invalid format)
- ph_sardaka_walking_002.jpg, true walking, output "standing" (wrong)
- ph_doggiebreath_lying_down_001.jpg, true lying down, output "lying down" (correct)

## Other jobs at the same check
B (2.5, us-central1), C (2.5, us-east1), D (2.5, europe-west1) still RUNNING; F (2.5-flash-lite) also pending/running.
With a winner in hand, the extra jobs are to be cancelled so they do not bill on completion.
