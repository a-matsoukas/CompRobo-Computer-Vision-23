import cv2
import numpy as np
import os
import mediapipe as mp


class keypointExtractor:
    """
    A class for extracting keypoint information from videos to use as training data.

    Attributes:
        mpHolistic: a mediapipe model that returns pose, face, and hand markers from a video feed in real time
        mpDrawing: a mediapipe helper class that visualizes results of a vision task
        DATAPATH: a string with the relative filepath to the top level data folder
        VIDEOPATH: a string with the filepath to the folder holding training videos
        videoProperties: a dict holding info for each video
        actions: a numpy array of text labels for each class of data
        sequenceLength: an int representing the number of frames to take for each video sample
    """

    def __init__(self, DATAPATH="./WLASLData", VIDEOPATH="./videos", videoProperties={}, actions=[], sequenceLength=30):
        """
        Initialize properties of instance of keypointExtractor.

        Args:
            DATAPATH: a string with the filepath to the top level folder that will hold data
                defaults to "./WLASLData"
            VIDEOPATH: a string the the filepath to the folder holding all of the training videos
                defaults to "./videos"
            videoProperties: a dict holding info for each video
                defaults to {}
            actions: a list of strings which are labels for each class of data
                defaults to []
            sequenceLength: an int representing the number of frames per data sample
                defaults to 30
        Returns:
            N/A
        """
        self.mpHolistic = mp.solutions.holistic
        self.mpDrawing = mp.solutions.drawing_utils

        self.DATAPATH = DATAPATH
        self.VIDEOPATH = VIDEOPATH
        self.videoProperties = videoProperties
        self.actions = np.array(actions)
        self.sequenceLength = sequenceLength

    def realTimeAnalysis(self, model):
        """
        Use a trained action detection model to predict sign language in real time.

        Args:
            model: the model attribute of a trainedModel class
        Returns:
            N/A
        """
        sequence = []
        sentence = []
        threshhold = .4

        cap = cv2.VideoCapture(0)

        # set mediapipe model
        with self.mpHolistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
            while cap.isOpened():

                # read from feed
                ret, frame = cap.read()

                # make detections from current frame
                image, results = self.mediapipeDetection(frame, holistic)

                # draw landmarks on frame
                self.drawLandmarks(image, results)

                # make predictions
                keypoints = self.extractKeypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]

                    # write text to screen
                    if (res[np.argmax(res)] > threshhold) and ((len(sentence) > 0 and self.actions[np.argmax(res)] != sentence[-1]) or (len(sentence) == 0)):
                        sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, " ".join(
                        sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # show to screen
                cv2.imshow("OpenCV Feed", image)

                # break condition
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def dataCapture(self):
        """
        Loop through videos in video folder, extract keypoint data from hands and face and save to file.

        Args:
            N/A
        Returns:
            N/A
        """
        # set mediapipe model
        with self.mpHolistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
            # loop through videos
            for videoFile in sorted(os.listdir(self.VIDEOPATH)):
                # extract id, action, instance id, and start/end frames
                videoID = videoFile[:-4]
                action = self.videoProperties[videoID]["gloss"]
                instanceID = self.videoProperties[videoID]["instance_id"]
                # starting at 1
                frameStart = self.videoProperties[videoID]["frame_start"]
                frameEnd = self.videoProperties[videoID]["frame_end"]

                cap = cv2.VideoCapture(os.path.join(self.VIDEOPATH, videoFile))

                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if frameEnd == -1:
                    frameEnd = frameCount

                # check if there are enough frames to collect data
                # incorporate a 5-frame buffer at the end due to issues with end of video not loading
                if (frameEnd - 5) - frameStart + 1 >= self.sequenceLength:
                    print(
                        f"Class: {action}, Instance: {instanceID}, Frames: {frameStart}:{frameEnd}, NumFrames:{(frameEnd - 5) - frameStart + 1}")
                    # get evenly spaced frames over the video
                    framesToAnalyze = np.linspace(
                        frameStart, frameEnd - 5, self.sequenceLength)

                    for frameNum in framesToAnalyze:

                        # set video to each frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frameNum) - 1)

                        # read from feed
                        ret, frame = cap.read()

                        if ret:
                            # make detections from current frame
                            image, results = self.mediapipeDetection(
                                frame, holistic)

                            # draw landmarks and show to screen
                            self.drawLandmarks(image, results)
                            cv2.imshow("Video", image)

                            # save frame keypoints
                            keypoints = self.extractKeypoints(results)
                            savePath = os.path.join(
                                self.DATAPATH, action, str(instanceID), str(int(frameNum)))
                            np.save(savePath, keypoints)

                        else:
                            print(f"\tMissed frame {int(frameNum)}")

                        # break condition
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                    cap.release()
                # if not enough frames, skip video
                else:
                    print(
                        f"SKIPPING - Class: {action}, Instance: {instanceID}, Frames: {frameStart}:{frameEnd}, NumFrames:{(frameEnd - 5) - frameStart + 1}")
                    cap.release

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
