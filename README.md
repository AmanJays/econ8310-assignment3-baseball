# Assignment 3
## Econ 8310 - Business Forecasting

For homework assignment 3, you will work with our baseball pitch data (available in Canvas).

- You must create a custom data loader as described in the first week of neural network lectures to load the baseball videos [2 points]
- You must create a working and trained neural network (any network focused on the baseball pitch videos will do) using only pytorch [2 points]
- You must store your weights and create an import script so that I can evaluate your model without training it [2 points]

Submit your forked repository URL on Canvas! :) I'll be manually grading this assignment.

Some checks you can make on your own:
- Can your custom loader import a new video or set of videos?
- Does your script train a neural network on the assigned data?
- Did your script save your model?
- Do you have separate code to import your model for use after training?

# Econ 8310: Assignment 3 - Baseball Detection

## Project Overview
This project implements a Faster R-CNN object detection model to identify baseballs in motion from video data. The model was trained using 286 annotated frames.

## Training Details
- **Architecture:** Faster R-CNN with a ResNet-50 backbone.
- **Optimization:** Stochastic Gradient Descent (SGD).
- **Learning Rate:** 0.001 (Optimized to prevent gradient explosion).
- **Hardware:** Trained on CPU for maximum stability on MacOS.
- **Epochs:** 3
- **Final Average Loss:** 0.3730

## How to Run
1. **Setup:** Install dependencies using `pip install -r requirements.txt`.
2. **Train:** Run `python3 assignment_script.py` to generate the `baseball_model.pt` file.
3. **Evaluate:** Run `python3 model_import.py` to run inference on a test video.

## Results
The model successfully detects baseballs with a peak confidence score of **89.8%**.