from keypointExtractor import keypointExtractor
from trainedModel import trainedModel


def main():
    """
    Call classes that collect data, train a model, and predict in real time.
    """
    extractor = keypointExtractor()
    modelInstance = trainedModel()

    extractor.realTimeAnalysis(modelInstance.model)


if __name__ == "__main__":
    main()
