from dotenv import load_dotenv
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with API key
MODEL = "text-embedding-3-small"
client = OpenAI(api_key=api_key)

# Extract text content from two PDF files
text1 = extract_text('sample1.pdf')
text2 = extract_text('sample2.pdf')

# Generate embeddings for the extracted texts using OpenAI API
vec1 = client.embeddings.create(
    model=MODEL,
    input=text1
)
embedding1 = np.array(vec1.data[0].embedding)

vec2 = client.embeddings.create(
    model=MODEL,
    input=text2
)
embedding2 = np.array(vec2.data[0].embedding)

# Compute cosine similarity between the two embeddings
similarity = cosine_similarity([embedding1], [embedding2])[0][0]
similarity_percentage = similarity * 100

print(f"\n{similarity_percentage:.2f}% of the texts match.")

# Interpret similarity score
if similarity_percentage > 90:
    print("The texts are very similar (possible plagiarism).\n")
elif similarity_percentage > 70:
    print("The texts are similar (possible paraphrasing).\n")
elif similarity_percentage > 50:
    print("The texts have some similarity (possible common topic).\n")
else:
    print("The texts are not similar (possible different topics).\n")
