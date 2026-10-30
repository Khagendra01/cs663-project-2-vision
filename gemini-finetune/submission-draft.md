We fine tuned Gemini on the dog behavior dataset from our Project 2. The model gets one dog image and has to say what the dog is doing. It can only answer with one of these four labels: sitting, standing, lying down, or walking. We used the same prompt in training and testing: "Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking."

Our dataset has 171 examples. The images are openly licensed dog photos and video frames from Wikimedia Commons (CC0, CC BY, and CC BY-SA), and we kept the license info in our dataset notes. I should be clear here: our first plan was to record our own dogs, but those recordings were not ready in time, so we used licensed stock images instead. We still checked every label by looking at the image. The split was done by source, so one source does not show up in two splits. The split is train 102, validation 26, and test 43. By class, there are 57 lying down, 42 standing, 37 walking, and 35 sitting. The images are in gs://cs663-dog-training-dog-data/images/ and the data files are train.jsonl, validation.jsonl, and test.jsonl in gs://cs663-dog-training-dog-data/datasets/. We dropped rolling over because it is a video behavior and one still image is not enough to judge it.

For tuning we used Google Cloud project cs663-dog-training and supervised fine tuning on Agent Platform. All runs used 1 epoch, learning rate multiplier 1.0, and adapter size 4.

Run 1 was gemini-3.5-flash, tuned name dog-training-gemini-v1, job 1708100736908263424, in us-central1. We sent it on Oct 29 at 11:56 PM PT.
Run 2 was gemini-2.5-flash, tuned name dog-training-gemini-25-v1, job 6332118877752786944, in us-central1. We sent it on Oct 30 at 7:40 AM PT because Run 1 was stuck in preparing for a long time.
Run 3 was gemini-2.5-flash, tuned name dog-training-gemini-25-e1, job 3019394375961542656, in us-east1. We sent it on Oct 30 at 9:07 AM PT.
Run 4 was gemini-2.5-flash, tuned name dog-training-gemini-25-eu1, job 333536336889774080, in europe-west1. We sent it on Oct 30 at 9:08 AM PT.

The problem is that none of these jobs had finished when we checked at about 10:35 AM PT. All four were still running and none of them showed an error. Run 1 had been going for about 10.5 hours by then. So we could not test the tuned model before the deadline. The base model results below are real and complete. When one tuning job finishes, we will run the same test on the tuned model.

For the test, we used the 43 held out test images and strict exact match scoring. The answer counted as correct only if it was exactly one label. We called the base models through the Agent Platform API using the global endpoint. In us-central1, the models gave 404 for this project, but they worked on global. One exception is gemini-2.5-flash. It gave 404 on both, so we could not test the untuned 2.5 base model directly. We are not guessing a score for that one.

The base gemini-3.5-flash model got 37 out of 43, which is 86.0%. We also tested gemini-3-flash-preview as a reference and it got 40 out of 43, which is 93.0%. The tuned model score is not available yet because the jobs were still running.

Per class, base gemini-3.5-flash got sitting 8 out of 9, standing 8 out of 11, walking 8 out of 9, and lying down 13 out of 14. For gemini-3-flash-preview, it was sitting 9 out of 9, standing 10 out of 11, walking 7 out of 9, and lying down 14 out of 14.

For the confusion matrix of base gemini-3.5-flash, the rows are the true label and the columns are the predicted label, in this order: sitting, standing, walking, lying down.
sitting: 8, 1, 0, 0
standing: 0, 8, 1, 0, and 2 more answers had the right word but in a sentence, so we counted them wrong
walking: 0, 1, 8, 0
lying down: 0, 0, 0, 13, and 1 more answer had the right words but in a sentence, so we counted it wrong

For gemini-3-flash-preview, the matrix was:
sitting: 9, 0, 0, 0
standing: 0, 10, 1, 0
walking: 0, 1, 7, 0, and 1 more answer had the right word but in a sentence, so we counted it wrong
lying down: 0, 0, 0, 14

A few examples from base gemini-3.5-flash:
1. ph_acabashi_standing_001.jpg, true label standing, model said "walking". This one is just wrong. The dog looks like it is mid step.
2. Test index 10, true label standing, model said "The dog is standing." The behavior is right, but the format is wrong.
3. Test index 11, true label standing, model said "The dog is **standing**." Same format problem.
4. Test index 39, true label lying down, model said "The dog is **lying down**." Same format problem.

So the base model is actually pretty good at the behavior part. Its main problem is format. It does not always give only the label like we asked. That is the main thing we expect fine tuning to fix.

Overall, we finished the dataset, the upload, the tuning setup, and the base model testing. The only missing part is the tuned model score, and that is missing because Google did not finish any tuning job in time. The dataset also has limits. We used stock images, not our own dogs, and sitting had the fewest examples. For the final Project 2 work, we plan to add our own dog recordings and more sitting examples.
