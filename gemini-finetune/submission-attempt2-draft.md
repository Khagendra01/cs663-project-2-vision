We fine tuned Gemini on the dog behavior dataset from our Project 2. The model gets one dog image and has to say what the dog is doing. It can only answer with one of these four labels: sitting, standing, lying down, or walking. We used the same prompt in training and testing: "Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking."

Our dataset has 171 examples. The images are openly licensed dog photos and video frames from Wikimedia Commons (CC0, CC BY, and CC BY-SA), and we kept the license info in our dataset notes. I should be clear here: our first plan was to record our own dogs, but those recordings were not ready in time, so we used licensed stock images instead. We still checked every label by looking at the image. The split was done by source, so one source does not show up in two splits. The split is train 102, validation 26, and test 43. By class, there are 57 lying down, 42 standing, 37 walking, and 35 sitting. The images are in gs://cs663-dog-training-dog-data/images/ and the data files are train.jsonl, validation.jsonl, and test.jsonl in gs://cs663-dog-training-dog-data/datasets/. We dropped rolling over because it is a video behavior and one still image is not enough to judge it.

For tuning we used Google Cloud project cs663-dog-training and supervised fine tuning on Agent Platform. All runs used 1 epoch, learning rate multiplier 1.0, and adapter size 4.

Run 1 was gemini-3.5-flash, tuned name dog-training-gemini-v1, job 1708100736908263424, in us-central1. We sent it on Oct 29 at 11:56 PM PT. This is the run that finished, and it is the one we tested. It took a long time, almost 17 hours. It finished on Oct 30 at 4:53 PM PT.
Run 2 was gemini-2.5-flash, tuned name dog-training-gemini-25-v1, job 6332118877752786944, in us-central1. We sent it on Oct 30 at 7:40 AM PT because Run 1 was stuck in preparing for a long time.
Run 3 was gemini-2.5-flash, tuned name dog-training-gemini-25-e1, job 3019394375961542656, in us-east1. We sent it on Oct 30 at 9:07 AM PT.
Run 4 was gemini-2.5-flash, tuned name dog-training-gemini-25-eu1, job 333536336889774080, in europe-west1. We sent it on Oct 30 at 9:08 AM PT.
We also sent one more run on gemini-2.5-flash-lite later that day. We tried gemini-2.0-flash too, but Google said that model is not supported for tuning. After Run 1 finished, we cancelled the other runs because we did not need them anymore.

One honest note: our first submission for this exercise went in at 10:51 AM on Oct 30, before any tuning job had finished, so it reported the tuned score as not ready yet. This version has the real tuned results, now that Run 1 is done.

For the test, we used the 43 held out test images and strict exact match scoring. The answer counted as correct only if it was exactly one label. We called the base models through the Agent Platform API using the global endpoint. In us-central1, the models gave 404 for this project, but they worked on global. One exception is gemini-2.5-flash. It gave 404 on both, so we could not test the untuned 2.5 base model directly. The tuned model also needed a small fix from our side: calling it by its model name gave 404, so we called it through its deployed endpoint, and that worked.

The results are like this. The tuned model got 38 out of 43, which is 88.4%. The base gemini-3.5-flash model got 37 out of 43, which is 86.0%. So tuning helped a little, one more correct image. We also tested gemini-3-flash-preview as a reference and it got 40 out of 43, which is 93.0%. If we count answers that had the right label inside a sentence as correct, then the tuned model and the base model both get 40 out of 43.

Per class, the tuned model got sitting 8 out of 9, standing 8 out of 11, walking 8 out of 9, and lying down 14 out of 14. The base model got sitting 8 out of 9, standing 8 out of 11, walking 8 out of 9, and lying down 13 out of 14. So the main gain was lying down, which became perfect after tuning.

For the confusion matrix of the tuned model, the rows are the true label and the columns are the predicted label, in this order: sitting, standing, walking, lying down.
sitting: 8, 0, 0, 0, and 1 more answer had the right word but in a sentence, so we counted it wrong
standing: 0, 8, 1, 0, and 2 more answers had the right word but in a sentence, so we counted them wrong
walking: 0, 1, 8, 0
lying down: 0, 0, 0, 14

For the base model, the matrix was:
sitting: 8, 1, 0, 0
standing: 0, 8, 1, 0, and 2 more answers had the right word but in a sentence, so we counted them wrong
walking: 0, 1, 8, 0
lying down: 0, 0, 0, 13, and 1 more answer had the right words but in a sentence, so we counted it wrong

A few examples from the tuned model:
1. ph_acabashi_standing_001.jpg, true label standing, model said "walking". This one is wrong. The base model got this one wrong too.
2. ph_sardaka_walking_002.jpg, true label walking, model said "standing". Wrong.
3. ph_f_standing_002.jpg, true label standing, model said "Based on the image, the dog is **standing**." The behavior is right, but the format is wrong.
4. ph_gazebo_sitting_001.jpg, true label sitting, model said "The dog is sitting." Same format problem.
5. ph_doggiebreath_lying_down_001.jpg, true label lying down, model said "lying down". Correct.

So our honest read is this: tuning helped a little. Accuracy went from 86.0% to 88.4%, and lying down became perfect. But tuning did not fully fix the format problem. The tuned model still gave 3 long answers instead of one word. One epoch was maybe not enough to make the format fully stable. And if we count the right word inside a sentence as correct, tuned and base are tied at 40 out of 43, so most of the strict score gap is about format, not about knowing the behavior.

Overall, we finished the dataset, the upload, the tuning, and the testing of both models. The dataset has limits. We used stock images, not our own dogs, and sitting had the fewest examples with only 35. For the final Project 2 work, we plan to add our own dog recordings and more sitting examples, and we may try more epochs to see if the format gets fully fixed.
