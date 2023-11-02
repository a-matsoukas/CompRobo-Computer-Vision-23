import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp


class keypointExtractor:
    """
    A class for extracting keypoint information from a video or camera feed to use as training data.

    Attributes:
        mpHolistic: a mediapipe model that returns pose, face, and hand markers from a video feed in real time
        mpDrawing: a mediapipe helper class that visualizes results of a vision task
        DATAPATH: a string with the relative filepath to the top level data folder
        nSequences: an int representing the number of video samples to take of each action
        sequenceLength: an int representing the number of frames to take for each video sample
    """

    def __init__(self, actions=np.array(["hello", "thank you", "I love you"]), nSequences=30, sequenceLength=30):
        """
        Initialize properties of instance of keypointExtractor.

        Args:
            actions: a numpy array of strings which are labels for each class of data
                defaults to ["hello", "thank you", "I love you"]
            nSequences: an int representing the number of data samples per class
                defaults to 30
            sequenceLength: an int representing the number of frames per data sample
                defaults to 30
        Returns:
            N/A
        """
        self.mpHolistic = mp.solutions.holistic
        self.mpDrawing = mp.solutions.drawing_utils

        self.DATAPATH = "./MPData"
        self.actions = actions
        self.nSequences = nSequences
        self.sequenceLength = sequenceLength

        self.makeDataDirectories()

    def videoCapture(self):
        """
        Open a CV2 video feed, extract keypoint data from hands and face and save to file.

        Args:
            N/A
        Returns:
            N/A
        """

        cap = cv2.VideoCapture(0)

        # set mediapipe model
        with self.mpHolistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
            for action in self.actions:
                for sequence in range(self.nSequences):
                    for frameNum in range(self.sequenceLength):

                        # read from feed
                        ret, frame = cap.read()

                        # make detections from current frame
                        image, results = self.mediapipeDetection(
                            frame, holistic)

                        # draw landmarks on frame
                        self.drawLandmarks(image, results)

                        # wait logic
                        if frameNum == 0:
                            cv2.putText(image, "STARTING COLLECTION", (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.imshow("OpenCV Feed", image)
                            cv2.waitKey(2000)
                        cv2.putText(image, f"Collecting frames for {action}. Video number {sequence}.", (
                            15, 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow("OpenCV Feed", image)

                        # save frame keypoints
                        keypoints = self.extractKeypoints(results)
                        savePath = os.path.join(
                            self.DATAPATH, action, str(sequence), str(frameNum))
                        np.save(savePath, keypoints)

                        # break condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return
            cap.release()
            cv2.destroyAllWindows()

    def mediapipeDetection(self, image, model):
        """
        Use MediaPipe to extract body keypoints from a given image.

        Args:
            image: a cv2 video capture frame in BGR color format
            model: the mediapipe model that will analyze the frame
        Returns:
            image: the same input image
            results: the results object returned by the model containing the frame's keypoints

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def drawLandmarks(self, image, results):
        """
        Annotate a given image with keypoints and connections.

        Args:
            image: a cv2 video capture frame
            results: keypoint data for the image 
        Returns:
            N/A
        """
        self.mpDrawing.draw_landmarks(image, results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION, self.mpDrawing.DrawingSpec(
            color=(80, 110, 10), thickness=1, circle_radius=1), self.mpDrawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        self.mpDrawing.draw_landmarks(image, results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS, self.mpDrawing.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4), self.mpDrawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        self.mpDrawing.draw_landmarks(image, results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS, self.mpDrawing.DrawingSpec(
            color=(121, 22, 76), thickness=2, circle_radius=4), self.mpDrawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        self.mpDrawing.draw_landmarks(image, results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS, self.mpDrawing.DrawingSpec(
            color=(245, 117, 66), thickness=2, circle_radius=4), self.mpDrawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def extractKeypoints(self, results):
        """
        Extract all keypoint info from the results of a frame and package into a 1-D numpy array.

        Args:
            results: keypoint data for a given frame
        Returns:
            a np array of size (1662,) holding pose, face, left hand, and right hand keypoint info
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose, face, lh, rh])

    def makeDataDirectories(self):
        """
        If not present, make directories to hold trianing data based on classes, num samples, and num frames.
        To be called when initializing keypointExtractor class.

        Args:
            N/A
        Returns:
            N/A
        """
        for action in self.actions:
            for sequence in range(self.nSequences):
                try:
                    os.makedirs(os.path.join(
                        self.DATAPATH, action, str(sequence)))
                except:
                    pass


extractor = keypointExtractor()
extractor.videoCapture()
