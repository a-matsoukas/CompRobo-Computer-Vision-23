import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


class trainedModel:
    """
    A class for training an LSTM for action detection given sequential keypoint data.

    Attributes:
        DATAPATH: a string with the relative filepath to the top level data folder
        actions: a numpy array of text labels for each class of data
        labelMap: a dict mapping each action to a unique integer from 0 to len(actions) - 1
        sequenceLength: an int representing the number of frames to take for each video sample
        X_train: a random subset of the total data used to train the model
        X_test: the remaining data used to test model's accuracy
        y_train: labels corresponding to the training data
        y_test: labals corresponding to the testing data
        model: the trained LSTM model that is used to make predictions
    """

    def __init__(self, DATAPATH="./WLASLData", actions=[], sequenceLength=30):
        """
        Initialize properties of instance of trainedModel.

        Args:
            DATAPATH: a string with the filepath to the top level folder that holds data
                defaults to "./WLASLData"
            actions: a list of strings which are labels for each class of data
                defaults to []
            sequenceLength: an int representing the number of frames per data sample
                defaults to 30
        Returns:
            N/A
        """
        self.DATAPATH = DATAPATH
        self.actions = np.array(actions)
        self.labelMap = {label: num for num, label in enumerate(self.actions)}

        self.sequenceLength = sequenceLength

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.generateTrainTestData()
        self.trainModel()

    def generateTrainTestData(self):
        """
        Given a filepath to a set of data, format and divide data into train and test data.

        Args:
            N/A
        Returns:
            N/A
        """
        # check if data has already been extracted
        if not os.path.exists("./XArray.npy") or not os.path.exists("./yArray.npy"):
            sequences, labels = [], []  # to hold sequential data and class labels
            for action in self.actions:
                print(f"Processing Action `{action}`")
                pathToAction = os.path.join(self.DATAPATH, action)

                # get list of instances for each action and sort numerically
                instances = os.listdir(pathToAction)
                instances.sort(key=lambda x: int(x))

                for instance in instances:
                    pathToActionInstance = os.path.join(
                        pathToAction, instance)

                    # get list of frames for each instance and sort numerically
                    frames = os.listdir(pathToActionInstance)
                    frames.sort(key=lambda x: int(x[:-4]))

                    # check if there is correct number of frames
                    if len(frames) == self.sequenceLength:
                        window = []
                        for frame in frames:
                            pathToActionInstanceFrame = os.path.join(
                                pathToActionInstance, frame)

                            # join frames together
                            res = np.load(pathToActionInstanceFrame)
                            window.append(res)

                        # append all frames to set of sequences
                        sequences.append(window)
                        # append corresponding label
                        labels.append(self.labelMap[action])
            # make into numpy arrays and make labels categorical
            X = np.array(sequences)
            y = np.array(to_categorical(labels).astype(int))
            # save data for future use
            np.save("./XArray.npy", X)
            np.save("./yArray.npy", y)
        # if data already exists, load it
        else:
            X = np.load("./XArray.npy")
            y = np.load("./yArray.npy")

        # split into train and test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=.4)

    def trainModel(self):
        """
        Initialize LSTM model and train on data if parameters are not saved.

        Args:
            N/A
        Returns:
            N/A
        """
        logDir = os.path.join("./Logs")
        tbCallback = TensorBoard(log_dir=logDir)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True,
                       activation="relu", input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.actions.shape[0], activation="softmax"))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=[
            "categorical_accuracy"])

        if os.path.exists("./trainedModel.keras"):
            model.load_weights("./trainedModel.keras")
        else:
            model.fit(self.X_train, self.y_train,
                      epochs=750, callbacks=[tbCallback])
            model.save("./trainedModel.keras")

        self.model = model

    def evaluateModel(self):
        """
        Print metrics evaluating model performance: confusion matrix and accuracy

        Args:
            N/A
        Returns:
            N/A
        """
        yHat = np.argmax(self.model.predict(self.X_test), axis=1).tolist()
        yTrue = np.argmax(self.y_test, axis=1).tolist()

        print(
            f"Confusion Matrix:\n {multilabel_confusion_matrix(yTrue, yHat)}")
        print(f"Accuracy Score:\n {accuracy_score(yTrue, yHat)}")
