from vllm import LLM, SamplingParams
import json
import sys
import torch
import re

filter_prompt = """
You are tasked with filtering texts so that scientists can help the world by analyzing important texts.
You should filter out texts with these following criteria:
1. A text should be filtered if it mainly consists of spoken or informal language
2. A text should be filtered if it mostly consists of lists instead of proper sentences
3. A text should be filtered if it is not at least four sentences long
4. A text should be filtered if it contains many headers, timestamps, links or other elements that are typical in online websites, but not in news articles or book chapters
5. A text should be filtered is it 
"""

def main():

    pass