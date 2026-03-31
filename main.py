import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is machine learning?",
    "What is deep learning?",
    "Explain neural networks",
    "What is NLP?",
    "Define artificial intelligence"
]

answers = [
    "Machine learning is a subset of AI that allows systems to learn from data.",
    "Deep learning is a subset of ML using neural networks with many layers.",
    "Neural networks are computing systems inspired by the human brain.",
    "NLP is Natural Language Processing, used to understand human language.",
    "Artificial Intelligence is the simulation of human intelligence in machines."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_answer(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    return answers[index]

print("Virtual Teaching Assistant (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = get_answer(user_input)
    print("Assistant:", response)
