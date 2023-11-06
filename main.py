from dataProcessor import dataProcessor
from keypointExtractor import keypointExtractor
from trainedModel import trainedModel


def main():
    """
    Call classes that collect data, train a model, and predict in real time.

    Args:
        N/A
    Returns:
        N/A
    """
    # initialize class to process data, create data directory, and access video info
    DATAPATH = "./WLASLData"  # where training/testing data will be saved
    # file where info about videos is stored in json format
    jsonFilepath = "./WLASL_v0.3.json"

    processor = dataProcessor(DATAPATH=DATAPATH, jsonFilepath=jsonFilepath)
    videoProperties = processor.videoIDInfoJSON
    actions = processor.actions

    # initialize class to extract datapoints from each video and save to each file
    VIDEOPATH = "./videos"  # folder where videos are stored
    sequenceLength = 30  # number of frames to collect per video

    extractor = keypointExtractor(DATAPATH=DATAPATH, VIDEOPATH=VIDEOPATH,
                                  videoProperties=videoProperties, actions=actions, sequenceLength=sequenceLength)

    # uncomment the following line if you don't have data in "./WLASLData"
    # extractor.dataCapture()

    # initialize class to train model based on data
    modelInstance = trainedModel(
        DATAPATH=DATAPATH, actions=actions, sequenceLength=sequenceLength)
    modelInstance.evaluateModel()

    # use the extractor to make real time predictions
    # extractor.realTimeAnalysis(modelInstance.model)


if __name__ == "__main__":
    main()
