from sentence_transformers import SentenceTransformer,util
import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import quote
faq = {
    # General
    "What is Python?": "Python is a high-level, interpreted programming language known for its readability and versatility.",
    "What is Python used for?": "Python is used for web development, data science, artificial intelligence, automation, scripting, and more.",
    "How do I check Python version?": "Use 'python --version' or 'python3 --version' in the terminal.",

    # Data Types
    "What are the basic data types in Python?": "int, float, str, bool, list, tuple, dict, set are the basic built-in types.",
    "What is a list in Python?": "A list is an ordered, mutable collection. Example: [1, 2, 3].",
    "What is a tuple in Python?": "A tuple is an ordered, immutable collection. Example: (1, 2, 3).",
    "What is a dictionary in Python?": "A dictionary stores key-value pairs. Example: {'a': 1, 'b': 2}.",
    "What is a set in Python?": "A set is an unordered collection of unique elements. Example: {1, 2, 3}.",
    "What is None in Python?": "None represents the absence of a value.",

    # Variables and Operators
    "How do I declare a variable in Python?": "Simply assign: x = 10. No need to declare type explicitly.",
    "What are Python operators?": "Arithmetic (+, -, *, /), comparison (==, !=, >, <), logical (and, or, not), assignment (=, +=, -=), and more.",

    # Strings
    "How do I create a string in Python?": "Use quotes: 'hello' or \"hello\".",
    "How do I concatenate strings in Python?": "Use + operator: 'Hello' + ' World'.",
    "How do I format strings in Python?": "Use f-strings: name = 'Tom'; print(f'Hello {name}').",

    # Control Flow
    "What is an if statement in Python?": "It allows conditional execution: if x > 0: print('Positive').",
    "What is a for loop in Python?": "It iterates over sequences: for i in [1, 2, 3]: print(i).",
    "What is a while loop in Python?": "It runs while condition is True: while x < 5: print(x).",

    # Functions
    "How do I define a function in Python?": "Use def keyword: def greet(): print('Hello').",
    "What is return in Python?": "It sends a value back from a function: def add(a,b): return a+b.",
    "What are default arguments in Python?": "Function parameters can have defaults: def greet(name='Guest').",

    # OOP
    "What is a class in Python?": "A class is a blueprint for objects: class Person: pass.",
    "What is an object in Python?": "An object is an instance of a class.",
    "What is inheritance in Python?": "It allows a class to inherit attributes and methods from another class.",
    "What is polymorphism in Python?": "It allows methods to be used interchangeably across different classes.",
    "What is encapsulation in Python?": "It restricts access to variables/methods using private (_var) or protected conventions.",

    # Modules and Packages
    "How do I import a module in Python?": "Use import keyword: import math.",
    "How do I install a package in Python?": "Use pip: pip install package_name.",
    "What is __init__.py file in Python?": "It marks a directory as a Python package.",

    # Errors and Exceptions
    "What are exceptions in Python?": "Errors detected during execution, like ZeroDivisionError, ValueError, etc.",
    "How do I handle exceptions in Python?": "Use try-except block: try: x=1/0; except ZeroDivisionError: print('Error').",
    "What is finally in Python exception handling?": "The finally block always executes, regardless of errors.",

    # File Handling
    "How do I read a file in Python?": "with open('file.txt','r') as f: data = f.read().",
    "How do I write to a file in Python?": "with open('file.txt','w') as f: f.write('Hello').",

    # Miscellaneous
    "What is indentation in Python?": "Indentation defines code blocks. Default is 4 spaces.",
    "What are Python comments?": "Use # for single-line, triple quotes for multi-line comments.",
    "What is PEP 8 in Python?": "PEP 8 is the style guide for writing clean Python code.",
    "What is Python interpreter?": "The program that executes Python code line by line.",
    "What are Python virtual environments?": "They isolate dependencies for different projects. Example: python -m venv env."
}

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions=list(faq.keys())
faq_embeddings=model.encode(faq_questions,convert_to_tensor=True)
class HybridPyChat:
    def __init__(self,curated_threshold=0.65,web_search_enabled=True):
        self.curated_threshold=curated_threshold
        self.web_search_enabled=web_search_enabled
        self.query_cache={}
        self.model=model
    def get_embedding(self,text):
        return self.model.encode(text,convert_to_tensor=True)
    def search_curated_faq(self,query):
        print("searching faq...")
        query_embedding=self.get_embedding(query)
        scores=util.cos_sim(query_embedding,faq_embeddings)
        best_idx=scores.argmax().item()
        best_score=scores[0][best_idx].item()
        print(f"   Best match: '{faq_questions[best_idx]}' (score: {best_score:.4f})")
        if best_score>=self.curated_threshold:
            print("match found")
            return {
                'answer':faq[faq_questions[best_idx]],
                'source':'curated faq',
                'confidence':best_score,
                'matched question':faq_questions[best_idx]
            }
        else:
            print(f"score {best_score} is below the threshold value {self.curated_threshold}")
            return None
    def extract_from_url(self,url):
        try:
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response=request.

