# Import the required libraries and dependencies
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score # Added for completeness, though not used in app directly
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import gradio as gr

# Attempt to load .env file for local development
if load_dotenv():
    print("Loaded .env file")
else:
    print(".env file not found, relying on Hugging Face Space secrets or preset environment variables.")

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"

# Store the API key in a variable.
# This will pick up the OPENAI_API_KEY from Space secrets if set,
# or from the .env file if loaded locally, or be None if not set anywhere.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found. The app will likely fail if it requires OpenAI.")
    # For a deployed app, you might want to raise an exception if the key is critical:
    # raise ValueError("OPENAI_API_KEY not set. Please set it as a secret in your Hugging Face Space.")
else:
    print(f"OPENAI_API_KEY loaded, length: {len(OPENAI_API_KEY)}")

# Set the column width to view the statments.
pd.set_option('max_colwidth', 200)

# Load the dataset.
# IMPORTANT: Ensure "Combined_Data.csv" is in the root of your Hugging Face Space.
try:
    df = pd.read_csv("Combined_Data.csv", index_col="Unnamed: 0")
except FileNotFoundError:
    print("Error: Combined_Data.csv not found. Please ensure it's uploaded to your Space.")
    # Handle the error, e.g., by exiting or using a default small DataFrame for app structure testing
    # For now, we'll let it proceed and potentially fail, which Gradio's show_error=True should help display.
    df = pd.DataFrame(columns=['statement', 'status']) # Minimal df to allow app to load structure

# Data Cleanup and Preparation
if not df.empty:
    df = df.dropna(subset=['statement']) # Ensure 'statement' column exists before dropping NA

    # Using the analyzer to determine the sentiment of each statement.
    statements = df['statement'].to_list()
    sentiment_col = [] # Renamed to avoid conflict
    score_col = []     # Renamed to avoid conflict
    analyzer = SentimentIntensityAnalyzer()
    for stmt in statements: # Renamed to avoid conflict
        statement_sentiment = analyzer.polarity_scores(str(stmt)) # Ensure statement is string
        if statement_sentiment['compound'] >= 0.05:
            sentiment_col.append("Positive")
        elif statement_sentiment['compound'] <= -0.05:
            sentiment_col.append("Negative")
        else:
            sentiment_col.append("Neutral")
        score_col.append(statement_sentiment['compound'])

    df['sentiment'] = sentiment_col
    df['score'] = score_col

    # Set the features variable.
    X = df['statement']
    # Set the target variables.
    y_status = df['status']
    y_sentiment = df['sentiment']

    # Split data into training and testing for status
    X_status_train, X_status_test, y_status_train, y_status_test = train_test_split(X, y_status, test_size=0.25, random_state=1)

    # Split data into training and testing for sentiment
    X_sentiment_train, X_sentiment_test, y_sentiment_train, y_sentiment_test = train_test_split(X, y_sentiment, test_size=0.30, random_state=1)

    # ML Model for Status
    status_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=None)),
        ('classifier', LinearSVC(dual='auto')) # Explicitly set dual
    ])
    status_pipeline.fit(X_status_train, y_status_train)

    # ML Model for Sentiment
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=None)),
        ('classifier', LinearSVC(dual='auto')) # Explicitly set dual
    ])
    sentiment_pipeline.fit(X_sentiment_train, y_sentiment_train)

else: # Handle case where df is empty (e.g. CSV not found)
    print("DataFrame is empty, using dummy pipelines.")
    # Create dummy pipelines to allow the app to load without actual model training
    status_pipeline = None
    sentiment_pipeline = None


# Initialize LLM (only if API key is present)
llm = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.3)
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        # llm will remain None, chatbot function should handle this

def mental_health_chatbot(statement_input): # Renamed input variable
    if not OPENAI_API_KEY or llm is None:
        return "Error: OpenAI API key not configured or LLM failed to initialize. Please check server logs."
    if status_pipeline is None or sentiment_pipeline is None:
        return "Error: ML models not trained. Please check server logs, likely Combined_Data.csv is missing."

    format_template = """
    You are a clinical psychologist. Answer only questions that would be relevant to mental health.
    If you don't know the answer, say you don't know
    If the human asks questions not related to mental health, remind them that your job is to help
    them understand their mental health status, and ask them for a question on that topic. If they ask a question which
    there is not enough information to answer, tell them you don't know and don't make up an
    answer.

    Question: {query_text}
    Answer:
    """ # Renamed placeholder

    prompt_template = PromptTemplate(
        input_variables=["query_text"], # Updated
        template=format_template
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Ensure statement_input is a list for predict method
    current_status = status_pipeline.predict([str(statement_input)])[0] # Get single prediction
    current_sentiment = sentiment_pipeline.predict([str(statement_input)])[0] # Get single prediction

    query_data = {"query_text":f'The statement from the user is:{statement_input}\\n The mental health status of the user is/has:{current_status}\\n The sentiment of the statement is:{current_sentiment}\\n Does the user require any assistance? If so what would you suggest?'}

    result = chain.invoke(query_data)
    return result["text"]

# Define Gradio interface
app_interface = gr.Interface( # Renamed variable to avoid conflict with 'app.py' module name
    fn=mental_health_chatbot,
    inputs=gr.Textbox(label="Enter your statement"),
    outputs=gr.Textbox(label="Mental Health Chatbot Response", show_copy_button=True)
)

# Launch the Gradio app
if __name__ == '__main__':
    # This check ensures app.launch() is only called when running app.py directly,
    # not when imported as a module (though less common for Gradio apps).
    # Hugging Face Spaces will typically run this.
    app_interface.launch(show_error=True) # share=False is default, show_error=True for debugging in Space
    # For local testing, you might want app_interface.launch(share=True) to get a public link
    # but for HF Spaces, share=True is often handled by the platform or can cause issues.
    # Defaulting to share=False (or not specifying it) is usually best for HF Spaces.
    # show_error=True is useful for seeing errors in the HF Space logs.
