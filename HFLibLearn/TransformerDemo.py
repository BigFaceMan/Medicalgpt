from transformers import pipeline
from loguru import logger

classifier = pipeline("zero-shot-classification")

output = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

logger.info(output)

