import os
import json
import pandas as pd
from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.llm import LLMChain
import random
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


SOURCE_DATA_PATH = "reviews_supplements.csv" 
SYNTHETIC_DATA_PATH = "synthetic_reviews.json"            
MEMORY_FILE_PATH = "memory.json"                           
NUM_SYNTHETIC_REVIEWS = 1000                              
CONTROL_FACTORS = {
    "length": ["short", "medium", "long"],
    "topic_diversity": ["specific", "diverse"],
    "language_complexity": ["simple", "complex"]
}
SEED = 42
random.seed(SEED)

def get_llm():
    """Returns an instance of the llama3 model with specified parameters."""
    return ChatOllama(model="llama3", temperature=0.7)  # Increased temperature for variability


def get_generation_prompt_template():
    system_message = SystemMessagePromptTemplate.from_template(
        """
        You are an AI model specialized in generating Amazon reviews for Supplements/Vitamins.
        Your task is to create synthetic reviews that mimic realistic human-written text.
        Ensure the reviews are original and not direct copies of existing reviews.
        Control the following factors in the generated reviews:
        - Length: {length}
        - Topic Diversity: {topic_diversity}
        - Language Complexity: {language_complexity}
        """
    )
    
    human_message = HumanMessagePromptTemplate.from_template(
        """
        Example Review:
        {example_review}
        
        Generate a new review based on the above example with the specified control factors. Only return the review and nothing else.
        """
    )
    
    return ChatPromptTemplate(
        input_variables=["length", "topic_diversity", "language_complexity", "example_review"],
        messages=[
            system_message,
            human_message,
        ],
    )

def create_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)


def load_source_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source data file not found at {path}")
    
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("Source dataset must contain a 'text' column.")
    
    return df


def select_control_factors():
    length = random.choice(CONTROL_FACTORS["length"])
    topic_diversity = random.choice(CONTROL_FACTORS["topic_diversity"])
    language_complexity = random.choice(CONTROL_FACTORS["language_complexity"])
    return {
        "length": length,
        "topic_diversity": topic_diversity,
        "language_complexity": language_complexity
    }


def generate_synthetic_review(chain, example_review, control_factors):
    prompt_input = {
        "length": control_factors["length"],
        "topic_diversity": control_factors["topic_diversity"],
        "language_complexity": control_factors["language_complexity"],
        "example_review": example_review
    }
    
    response = chain.invoke(prompt_input)
    print(response)
    synthetic_review = response['text'].strip()
    
    # Optional: Post-process the synthetic review (e.g., remove unwanted characters)
    synthetic_review = re.sub(r'\s+', ' ', synthetic_review)
    
    return synthetic_review

def evaluate_synthetic_data(source_reviews, synthetic_reviews):

    evaluation = {}
    source_reviews = [str(i) for i in source_reviews]
    synthetic_reviews = [str(i) for i in synthetic_reviews]
    # Average Length
    source_lengths = [len(str(review).split()) for review in source_reviews]
    synthetic_lengths = [len(str(review).split()) for review in synthetic_reviews]
    evaluation['average_length_source'] = sum(source_lengths) / len(source_lengths)
    evaluation['average_length_synthetic'] = sum(synthetic_lengths) / len(synthetic_lengths)

    # Sentiment Analysis
    source_sentiments = [TextBlob(str(review)).sentiment.polarity for review in source_reviews]
    synthetic_sentiments = [TextBlob(str(review)).sentiment.polarity for review in synthetic_reviews]
    evaluation['average_sentiment_source'] = sum(source_sentiments) / len(source_sentiments)
    evaluation['average_sentiment_synthetic'] = sum(synthetic_sentiments) / len(synthetic_sentiments)

    # Plotting Distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(source_lengths, color='blue', label='Source', kde=True, stat="density", bins=30)
    sns.histplot(synthetic_lengths, color='orange', label='Synthetic', kde=True, stat="density", bins=30)
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(source_sentiments, color='blue', label='Source', kde=True, stat="density", bins=30)
    sns.histplot(synthetic_sentiments, color='orange', label='Synthetic', kde=True, stat="density", bins=30)
    plt.title('Sentiment Polarity Distribution')
    plt.xlabel('Polarity')
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.close()

    return evaluation

def ensure_non_replication(source_reviews, synthetic_reviews):
    source_reviews = [str(i) for i in source_reviews]
    synthetic_reviews = [str(i) for i in synthetic_reviews]
    source_set = set([review.lower() for review in source_reviews])
    for review in synthetic_reviews:
        if review.lower() in source_set:
            return False
    return True

def main():
    
    # Step 1: Load Source Data
    print("Loading source dataset...")
    source_df = load_source_data(SOURCE_DATA_PATH)
    source_reviews = source_df['text'].tolist()
    print(f"Loaded {len(source_reviews)} source reviews.")
    
    # Step 2: Initialize LLM and Chain
    print("Initializing language model...")
    llm = get_llm()
    prompt = get_generation_prompt_template()
    chain = create_chain(llm, prompt)
    
    # Step 3: Generate Synthetic Reviews
    print(f"Generating {NUM_SYNTHETIC_REVIEWS} synthetic reviews...")
    synthetic_reviews = []
    for i in range(NUM_SYNTHETIC_REVIEWS):
        # Select a random example review
        example_review = random.choice(source_reviews)
        
        # Select control factors
        control_factors = select_control_factors()
        
        # Generate synthetic review
        synthetic_review = generate_synthetic_review(chain, example_review, control_factors)
        
        synthetic_reviews.append({
            "original_review": example_review,
            "synthetic_review": synthetic_review,
            "control_factors": control_factors
        })
        
        if (i + 1) % 100 == 0:
            print(f"{i + 1} synthetic reviews generated.")
    
    # Step 4: Save Synthetic Dataset
    print("Saving synthetic dataset...")
    with open(SYNTHETIC_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(synthetic_reviews, f, ensure_ascii=False, indent=4)
    print(f"Synthetic dataset saved to {SYNTHETIC_DATA_PATH}.")
    with open('synthetic_reviews.json', 'r') as file:
        synthetic_reviews = json.load(file)

    # Step 5: Evaluation
    print("Evaluating synthetic dataset...")
    synthetic_texts = [item["synthetic_review"] for item in synthetic_reviews]
    
    evaluation_metrics = evaluate_synthetic_data(source_reviews, synthetic_texts)
    print("Evaluation Metrics:")
    for key, value in evaluation_metrics.items():
        print(f"{key}: {value}")
    print("Evaluation plots saved as 'evaluation_metrics.png'.")
    
    # Step 6: Ensure Non-Replication
    print("Checking for replication...")
    is_unique = ensure_non_replication(source_reviews, synthetic_texts)
    if is_unique:
        print("Success: No synthetic reviews are exact replicas of source reviews.")
    else:
        print("Warning: Some synthetic reviews replicate source reviews.")
    
    print("Synthetic data generation completed.")

if __name__ == "__main__":
    main()
