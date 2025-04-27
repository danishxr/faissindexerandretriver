from google import genai
from google.genai.types import EmbedContentConfig

client = genai.Client()
response = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents=[
        "How do I get a driver's license/learner's permit?",
        "How do I renew my driver's license?",
        "How do I change my address on my driver's license?",
    ],
    config=EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",  # Optional
        output_dimensionality=768,  # Optional
        title="Driver's License",  # Optional
    ),
)
print(response)
