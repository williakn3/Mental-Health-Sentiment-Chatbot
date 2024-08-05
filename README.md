# Mental-Health-Sentiment-Chatbot
This project aims to provide insights into users' mental health states based on their textual statements. By leveraging machine learning and natural language processing techniques, we can predict and analyze mental health statuses to offer valuable support and guidance.


References - Sarkar, Suchintika (2024), "Sentiment Analysis for Mental Health", Kaggle.

Dependencies
Pandas

MatPlotLib

pyplot
VADER Sentiment

SentimentIntensityAnalyzer
Scikit-Learn

train_test_split
Pipeline
TfidVectorizer
LinearSVC
confusion_matrix
classification_report
accuracy_score
dotenv

load_dotenv
os

LangChain

ChatOpenAI
PromptTemplate
ChatPromptTemplate
LLMChain
Gradio

Data Preprocessing:

ating an interactive user interface where users can input their statements and receive a response from the chatbot which will provide suggestions on how to manage any mental health issues.
Preprocessing
The dataset has 53403 rows with columns containing a statement and the mental health status associated with the statement. It contained 362 rows where the statement was null. After dropping rows with null values, the total number of entries is 52681. Each statement was analyzed with the Sentiment Intensity Analyzer from VADER Sentiment. The analyzer determines whether a statement is positive, negative, or neutral in sentiment. Two new columns were added that show the sentiment and the polarity score associated with the statement. The status column was set as the target variable and the data was split into training and testing data. The test size was 25% for y_status and 30% for y_sentiment.

ML Model
The ML Model is a pipeline of TfidVectorizer and Support Vector Classifier (SVC). When using the model for Y_status_train, when analyzed for different mental health types, identified seven common conditions. These, from most to least common, are Anxiety, Bipolar Disorder, Depression, Normal Thoughts, Personality Disorder, Stress, and Suicidal Tendencies. The model achieved an overall accuracy of 0.78. For the ML Model Y_sentiment_train, The overall accuracy of this sentiment analysis was 0.82. A Pipeline that used English stopwords did not produce higher accuracies.

Open AI
This system represents a sophisticated, multi-stage approach to Artificial Intelligence merged with mental health support. It combines prompt engineering, natural language processing, sentiment analysis, and large language model inference to provide contextually aware, empathetic responses to users' mental health queries. The system's modular design allows for easy updates and improvements allowing for the enhancement of the status and sentiment pipelines with more advanced models, and the expansion of the prompt template to cover more scenarios, or fine-tune the language model on mental health-specific data. While this system shows great promise in providing scalable, accessible mental health support, it's crucial to emphasize that it should complement, not replace, professional mental health services. Future developments could include integration with human oversight, expansion of the knowledge base, and implementation of more robust safeguards and ethical guidelines.

Gradio App
How It Works
User Interface: The Gradio app features two textboxes. Users can enter their statements in the input textbox, and the chatbot's reponse to the statement will be displayed in the output textbox.
Prediction Process: The app processes user inputs through the trained linear SVC model and provides insights into the user's mental state.
Model Integration: The app combines the power of the trained machine learning model with OpenAI's language model prompts to deliver accurate and helpful mental health predictions.
Usage
Launch the Gradio App:
Run the Gradio app to start the local server.
Input Statement:
Enter your statement about your mental state in the provided textbox.
Receive Prediction:
View the predicted mental health status in the output textbox.
Example Statements
"I have been feeling very anxious and stressed lately."
"I am experiencing severe depression and can't find motivation."
"I feel normal and haven't had any mental health issues recently."
"I've had suicidal thoughts and feel hopeless."
"I'm struggling with bipolar disorder and my mood swings are intense."
"I don't know what I'm doing with my life, and I feel lost and alone."
"I don't know if I have a mental illness. I just have trouble expressing myself and need help."
"I'm super happy."
Chatbot Responses
"Yes, it sounds like the user may benefit from some assistance in managing their anxiety and stress. I would suggest seeking support from a mental health professional, such as a therapist, who can help explore the underlying causes of the anxiety and provide coping strategies to manage it. Additonally, practicing relaxation techniques, mindfulness, and engaging in regular physical activity can be helpful in reducing anxity and stress."

"Yes, it sounds like the user is experiencing severe depression and a lack of motivation. I would suggest seeking help from a mental health professional, such as a therapist or psychiatrist, to discuss treatment options such as therapy, medication, or other interventions that may help improve their symptoms. It's important to reach out for support and not try to handle these feelings alone."

"It's great to hear that you feel normal and haven't had any recent mental health issues. However, since you mentioned having bipolar disorder, it's important to continue monitoring your mental health and seeking support when needed. I would suggest staying in touch with your mental health provider, practicing self-care, and reaching out for help if you notice any changes in your mood or behavior. It's always good to have a support system in place."

"Yes, the user definitely requires assistance. I would suggest that they seek help from a mental health professional, such as a therapist or psychiatrist, to address their suicidal thoughts and feelings of hopelessness. It's important to take these thoughts seriously and get the support needed to work through them."

"Yes, it sounds like the user could benefit from seeking assistance from a mental health professional, such as a psychiatrist or therapist, who can help with managing the intense mood swings associated with bipolar disorder. It's important to have a proper diagnosis and treatment plan in place to help manage symptoms and improve overall well-being. I would suggest reaching out to a mental health provider for further evaluation and support."

"Yes, it sounds like the user may be experiencing symptoms of depression. I would suggest seeking help from a mental health professional, such as a therapist or psychologist, to explore these feelings further and develop coping strategies. It's important to remember that you are not alone and there are resources available to support you."

"Yes, it sounds like the user may benefit from seeking help from a mental health professional, such as a therapist or psychologist, to further explore their difficulties with expressing themselves and determine if they may have a mental illness such as bipolar disorder. I would suggest reaching out to a mental health provider for an evaluation and to discuss treatment options. It's important to prioritize your mental health and seek support when needed."

"It's great to hear that you're feeling super happy! It sounds like you are in a positive state of mind. If you feel ever overwhelmed or need support, it's always helpful to talk to a mental health professional. Is there anything specific, you would like to discuss or ask about related to your mental health?

Purpose
The goal of this project is to offer a tool that can help users gain insights into their mental health by analyzing their statements. This can provide a starting point for seeking further support or understanding one's mental state better.
