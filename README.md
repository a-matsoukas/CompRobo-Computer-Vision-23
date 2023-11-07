# CompRobo-Computer-Vision-23

Course: A Computational Introduction to Robotics, Fall 2023

Professor: Paul Ruvolo

## Project Goal

The goal of this project was to explore computer vision within the context of sign language translation. I wanted to focus on real-time sign language translation, as opposed to doing image recognition on static photos; this is effectively action recognition, where all of the actions are words in sign language.

Part of the framing of this project was to learn at a high level and implement a new machine learning algorithm. I took ML a couple years ago, where I worked with multilayer perceptrons (MLPs), generative adversarial networks (GANs), and convolutional neural networks (CNNs). For this project, I implemented a long short-term memory (LSTM) algorithm, which differs from all of the above models in the sense that it is a type of recurrent neural network (RNN), and thus has “memory” and can make decisions based on sequential data, which is good for analyzing a real-time feed of data.

Additionally, I planned to do some reading about where this kind of tool is already being used, its benefits, and its negatives. I see this as a tool with a lot of potential to assist communication between those who use sign language and those who do not. I aimed to look into the social implications of this technology and what should be done to make sure that it is implemented equitably.

As a stretch goal, I planned to learn at a low level how LSTMs work, such that I can write out and/or describe the math and implement the model from scratch.

As a minimum viable product (MVP) for this project, I planned to follow along and implement the sign language recognition algorithm outlined in [this YouTube tutorial](https://www.youtube.com/watch?v=doDUihpj6ro). I was successful in implementing the MVP early on in the project, but I got hung on on implementation details and model-training quirks when making improvements to the algorithm; as a result, I was not quite able to explore social context or learn how the model works at a low level, as outlined by my strech goals.

## Implementation Details

### Dependencies

- TensorFlow
- OpenCV-Python
- MediaPipe
- scikit-learn
- Matplotlib
- [Files in this Folder](https://drive.google.com/drive/folders/1j_HCrhukWvCwATvi8vavOUxSfb0K09kA?usp=sharing)

**How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.**

## Design Decisions

**Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.**

## Challenges

**What if any challenges did you face along the way?**

## Future Improvements

**What would you do to improve your project if you had more time?**

## Lessons Learned

**Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.**

## Resources Used

- [Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model](https://www.youtube.com/watch?v=doDUihpj6ro)
- [WLASL (World Level American Sign Language) Video](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=WLASL_v0.3.json)
  - 12k processed videos of Word-Level American Sign Language glossary performance
