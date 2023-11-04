import os
import json


class dataProcessor:
    """
    A class for creating directories to store keypoints from training videos.

    Attributes:
        DATAPATH: a string that is the filepath to the folder that will hold training data
        dataJSON: a string that is the filepath to the json file holding info about each video
        videoIDInfoJSON: a dict that pairs each video id to dict of attributes

    Methods:
        makeIDClassDict
        makeDataDirectories
    """

    def __init__(self, DATAPATH="./WLASLData", jsonFilepath="./WLASL_v0.3.json"):
        """
        Initialize properties of instance of dataProcessor.

        Args:
            DATAPATH: a string indicating where data will be saved to generate folders
            jsonFilepath: a string indicating where info about each video is
        Returns:
            N/A
        """
        self.DATAPATH = DATAPATH
        with open(jsonFilepath) as jsonFile:
            self.dataJSON = json.load(jsonFile)

        self.videoIDInfoJSON = {}

        self.makeIDClassDict()
        self.makeDataDirectories()

    def makeIDClassDict(self):
        """
        Create a dict that rearanges info in the dataJSON file to enable indexing by video_id.
        To be called when initializing dataProcessor class.

        Args:
            N/A
        Returns:
            N/A
        """
        for sign in self.dataJSON:
            for instance in sign["instances"]:
                self.videoIDInfoJSON[instance["video_id"]] = {"gloss": sign["gloss"],
                                                              "bbox": instance["bbox"],
                                                              "fps": instance["fps"],
                                                              "frame_end": instance["frame_end"],
                                                              "frame_start": instance["frame_start"],
                                                              "instance_id": instance["instance_id"],
                                                              "signer_id": instance["signer_id"],
                                                              "source": instance["source"],
                                                              "split": instance["split"],
                                                              "url": instance["url"],
                                                              "variation_id": instance["variation_id"]}

    def makeDataDirectories(self):
        """
        If not present, make directories to hold training data based on available video data.
        To be called when initializing dataProcessor class.

        Args:
            N/A
        Returns:
            N/A
        """
        for videoID in self.videoIDInfoJSON:
            try:
                os.makedirs(os.path.join(self.DATAPATH, self.videoIDInfoJSON[videoID]["gloss"], str(
                    self.videoIDInfoJSON[videoID]["instance_id"])))
            except:
                pass


processor = dataProcessor()
