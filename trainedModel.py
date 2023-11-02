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
        nSequences: an int representing the number of video samples to take of each action
        sequenceLength: an int representing the number of frames to take for each video sample
        X_train: a random subset of the total data used to train the model
        X_test: the remaining data used to test model's accuracy
        y_train: labels corresponding to the training data
        y_test: labals corresponding to the testing data
        model: the trained LSTM model that is used to make predictions
    """

    def __init__(self, DATAPATH="./MPData", actions=np.array(["hello", "thank you", "I love you"]), nSequences=30, sequenceLength=30):
        """
        Initialize properties of instance of trainedModel.

        Args:
            DATAPATH: a string with the filepath to the top level folder that holds data
            actions: a numpy array of strings which are labels for each class of data
                defaults to ["hello", "thank you", "I love you"]
            nSequences: an int representing the number of data samples per class
                defaults to 30
            sequenceLength: an int representing the number of frames per data sample
                defaults to 30
        Returns:
            N/A
        """
        self.DATAPATH = DATAPATH
        self.actions = actions
        self.labelMap = {label: num for num, label in enumerate(self.actions)}

        self.nSequences = nSequences
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
        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(self.nSequences):
                window = []
                for frameNum in range(self.sequenceLength):
                    res = np.load(os.path.join(self.DATAPATH,
                                  action, str(sequence), f"{frameNum}.npy"))
                    window.append(res)
                sequences.append(window)
                labels.append(self.labelMap[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=.05)

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

        if os.path.exists("./trainedModel.h5"):
            model.load_weights("./trainedModel.h5")
        else:
            model.fit(self.X_train, self.y_train,
                      epochs=650, callbacks=[tbCallback])
            model.save("./trainedModel.h5")

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
