import dspy
import os
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv
from dspy.teleprompt import MIPRO
import json
import time
import random
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure DSPy with OpenAI LLM
llm = dspy.OpenAI(
    model='gpt-3.5-turbo',
    api_key="sk-proj-",
    max_tokens=2000
)
dspy.settings.configure(lm=llm)

# Replace JinaReaderWebSearch with PerplexityWebSearch
class PerplexityWebSearch(dspy.Module):
    def __init__(self, num_results=5, max_retries=3):
        super().__init__()
        self.num_results = num_results
        self.max_retries = max_retries
        self.api_key = "pplx-"
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
        logging.info(f"Perplexity API key loaded: {self.api_key[:5]}...{self.api_key[-5:]}")

    async def fetch_content(self, query):
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    logging.info(f"Attempting Perplexity API call (attempt {attempt + 1})")
                    async with session.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-sonar-small-128k-online",
                            "messages": [{"role": "user", "content": query}],
                            "max_tokens": 1024
                        }
                    ) as response:
                        logging.info(f"API response status: {response.status}")
                        response.raise_for_status()
                        data = await response.json()
                        logging.info(f"API response data: {data}")
                        content = data['choices'][0]['message']['content']
                        logging.info(f"Perplexity API response: {content[:200]}...")
                        return [content]
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        logging.error(f"All {self.max_retries} attempts failed for query '{query}'")
                        return [f"Error: Unable to fetch content from Perplexity API. Details: {str(e)}"]
                    await asyncio.sleep(1)

    def forward(self, query):
        logging.info(f"PerplexityWebSearch forward method called with query: {query}")
        loop = asyncio.get_event_loop()
        passages = loop.run_until_complete(self.fetch_content(query))
        logging.info(f"Fetched {len(passages)} passages")
        if not passages:
            logging.warning("No passages found, using placeholder")
            passages = ["No relevant content found for the given query."]
        logging.info(f"Returning passages: {passages[:2]}...")
        return dspy.Prediction(passages=passages)

# Update WeKnowRAG to use PerplexityWebSearch
class WeKnowRAG(dspy.Module):
    def __init__(self, num_passages=1):
        super().__init__()
        self.retrieve_web = PerplexityWebSearch(num_results=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        self.teleprompter = MIPRO(
            metric=custom_evaluation_metric,
            num_candidates=10,
            init_temperature=1.0,
            verbose=True
        )
        self.total_tokens = 0
        self.iteration_count = 0

    def forward(self, question):
        logging.info(f"WeKnowRAG forward method called with question: {question}")
        web_context = self.retrieve_web(question)
        logging.info(f"Retrieved web context: {web_context.passages[:2]}...")
        context = " ".join(web_context.passages)
        logging.info(f"Full context length: {len(context)} characters")
        
        # Estimate tokens for context and question
        context_tokens = len(context.split())
        question_tokens = len(question.split())
        
        prediction = self.generate_answer(context=context, question=question)
        logging.info(f"Generated answer: {prediction.answer}")
        
        # Estimate tokens for answer
        answer_tokens = len(prediction.answer.split())
        
        # Update token counts
        iteration_tokens = context_tokens + question_tokens + answer_tokens
        self.total_tokens += iteration_tokens
        self.iteration_count += 1
        
        logging.info(f"Iteration {self.iteration_count} tokens: {iteration_tokens}")
        logging.info(f"Total tokens so far: {self.total_tokens}")
        
        return dspy.Prediction(context=context, answer=prediction.answer, 
                               iteration_tokens=iteration_tokens, total_tokens=self.total_tokens)

    def compile_and_run(self):
        # Compile the WeKnow-RAG program with the example data
        compiled_weknow_rag = self.teleprompter.compile(self, trainset=trainset, valset=valset)

        # Function to answer the question using the WeKnow-RAG system
        def ask_weknow_rag(question):
            try:
                prediction = compiled_weknow_rag(question)
                print(f"Question: {question}")
                print(f"Predicted Answer: {prediction.answer}")
                print(f"Context length: {len(prediction.context)} characters")
                print(f"Context preview: {prediction.context[:200]}...")
                print(f"Tokens used in this iteration: {prediction.iteration_tokens}")
                print(f"Total tokens used: {prediction.total_tokens}")
            except Exception as e:
                logging.error(f"Error occurred while processing question: {e}")
                print(f"An error occurred: {e}")

        # Run the question
        ask_weknow_rag(my_question)

# Update the custom evaluation metric
def custom_evaluation_metric(example, prediction, trace=None):
    if not prediction.answer:
        return 0.0
    
    # Exact match score (keeping this from the original function)
    exact_match_score = dspy.evaluate.answer_exact_match(example, prediction)
    
    # TF-IDF based similarity score
    vectorizer = TfidfVectorizer().fit_transform([example.answer, prediction.answer])
    cosine_sim = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    
    # Check for key phrases (adjust these based on your specific use case)
    key_phrases = ["machine learning", "artificial intelligence", "data analysis", "predictive models"]
    phrase_score = sum(phrase.lower() in prediction.answer.lower() for phrase in key_phrases) / len(key_phrases)
    
    # Length ratio score (penalize answers that are too short or too long)
    len_ratio = min(len(prediction.answer) / len(example.answer), len(example.answer) / len(prediction.answer))
    
    # Combine scores (adjust weights as needed)
    final_score = (
        0.3 * exact_match_score +
        0.4 * cosine_sim +
        0.2 * phrase_score +
        0.1 * len_ratio
    ) * 100  # Scale to 0-100
    
    logging.info(f"Evaluation scores - Exact match: {exact_match_score}, Cosine similarity: {cosine_sim}, "
                 f"Phrase score: {phrase_score}, Length ratio: {len_ratio}, Final: {final_score}")
    
    return final_score

# Example question
my_question = "What are the latest advancements in AI for personalized medicine?"

# Create some example data for training and validation
trainset = [
    dspy.Example(
        question="What is machine learning?",
        answer="Machine learning is a branch of artificial intelligence..."
    ).with_inputs("question"),
    dspy.Example(
        question="How does DNA sequencing work?",
        answer="DNA sequencing is the process of determining the nucleic acid sequence..."
    ).with_inputs("question"),
]

valset = [
    dspy.Example(
        question="What are the applications of blockchain in healthcare?",
        answer="Blockchain in healthcare can be used for secure medical records..."
    ).with_inputs("question"),
]

def compile_and_run():
    weknow_rag = WeKnowRAG()

    trial_count = 0
    total_tokens_across_trials = 0

    def token_tracking_metric(example, prediction, trace=None):
        nonlocal trial_count, total_tokens_across_trials
        trial_count += 1
        total_tokens_across_trials += prediction.iteration_tokens
        print(f"Trial {trial_count}")
        print(f"Tokens used in this trial: {prediction.iteration_tokens}")
        print(f"Total tokens used across all trials: {total_tokens_across_trials}")
        print("-" * 40)
        
        # Call the original metric function and return its result
        return custom_evaluation_metric(example, prediction, trace)

    teleprompter = MIPRO(
        metric=token_tracking_metric,
        num_candidates=10,
        init_temperature=1.0,
        verbose=True
    )

    compiled_weknow_rag = teleprompter.compile(
        weknow_rag,
        trainset=trainset,
        num_trials=50,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        eval_kwargs=dict(num_threads=1, display_progress=True, display_table=0),
        view_data=True,
        view_examples=True
    )

    print(f"Compilation complete. Total tokens used across all trials: {total_tokens_across_trials}")

    def ask_weknow_rag(question):
        try:
            prediction = compiled_weknow_rag(question)
            print(f"\nQuestion: {question}")
            print(f"Predicted Answer: {prediction.answer}")
            print(f"Context length: {len(prediction.context)} characters")
            print(f"Context preview: {prediction.context[:200]}...")
            print(f"Tokens used in this query: {prediction.iteration_tokens}")
            nonlocal total_tokens_across_trials
            total_tokens_across_trials += prediction.iteration_tokens
            print(f"Total tokens used (including compilation): {total_tokens_across_trials}")
        except Exception as e:
            logging.error(f"Error occurred while processing question: {e}")
            print(f"An error occurred: {e}")

    # Interactive question input loop
    while True:
        user_question = input("\nEnter your question (or 'quit' to exit): ").strip()
        if user_question.lower() == 'quit':
            break
        ask_weknow_rag(user_question)

def test_perplexity_api_sync():
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("PERPLEXITY_API_KEY not found in environment variables")
        return
    
    print(f"Perplexity API key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "max_tokens": 1024
            }
        )
        print(f"API response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API response data: {data}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Test Perplexity API
    test_perplexity_api_sync()

    # Run WeKnowRAG with interactive question input
    compile_and_run()

if __name__ == "__main__":
    main()
