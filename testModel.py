# Test Code... need to update path
from transformers import pipeline
PATH = "C:\\Users\\PC\\code\\githubProjects\\sentimentAnalysisTwitter\\"
sentiment_model = pipeline("text-classification", model = PATH + "trainedAlgorithm\\checkpoint-6250\\")

print(sentiment_model(["I love this move", "This movie sucks!"]))