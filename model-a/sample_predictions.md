# Model A - 10 sample test predictions

From the held-out v4 test set (235 images, evaluated once). Picked as a spread across all 4 classes, 7 correct and 3 misses - the misses are included on purpose; overall test accuracy is 74.9% (176/235).

| # | file | source | true | predicted | confidence | result |
|---|---|---|---|---|---|---|
| 1 | ph_presto808_sitting_001.jpg | ph_presto808 | sitting | sitting | 0.999 | correct |
| 2 | ph_evnerd_sitting_001.jpg | ph_evnerd | sitting | sitting | 0.999 | correct |
| 3 | ph_jason_nlw_sitting_001.jpg | ph_jason_nlw | sitting | walking | 0.991 | **MISS** |
| 4 | ph_adventurous36_standing_002.jpg | ph_adventurous36 | standing | standing | 0.999 | correct |
| 5 | ph_basile_morin_standing_001.jpg | ph_basile_morin | standing | lying_down | 0.968 | **MISS** |
| 6 | ph_inugami_bargho_lying_down_002.jpg | ph_inugami_bargho | lying_down | lying_down | 1.000 | correct |
| 7 | ph_nicole_qowens_lying_down_001.jpg | ph_nicole_qowens | lying_down | lying_down | 1.000 | correct |
| 8 | ph_chyah_walking_002.jpg | ph_chyah | walking | walking | 1.000 | correct |
| 9 | ph_chyah_walking_003.jpg | ph_chyah | walking | walking | 1.000 | correct |
| 10 | ph_wolfmann_walking_001.jpg | ph_wolfmann | walking | sitting | 0.983 | **MISS** |

Pattern in the misses (matches the confusion matrix): standing dogs get called sitting or walking, and walking dogs get called sitting/standing - posture at a single frame is genuinely ambiguous between those pairs; lying_down is the most reliable class.
