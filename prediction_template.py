from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

checkpoint = '/models/fine_tuned_distilbert'  # Path to model goes here.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Tokenizer can be adjusted if needed.

pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

df = pd.read_csv('questions.csv', encoding='latin1')  # Include file name and proper encoding here.

# Create lists to store information for new columns.
predictions = []
scores = []

# Predict each question and add predictions to column lists.
for q in df['question']:  # Replace 'question' with the proper column name.
    q = str(q)  # As a precaution.
    pred_dict = pipe(q)[0]  # Gives dictionary of label and score.
    label = pred_dict['label']

    # This part is optional to make the labels more readable.
    # Replace names/shorthands as necessary.
    if label == 'LABEL_0':
        predictions.append('D')
    elif label == 'LABEL_1':
        predictions.append('R')
    else:
        predictions.append('N')  # In case of an invalid label.
    # Entire preceding part can be replaced by predictions.append(label)

    scores.append(pred_dict['score'])

df['prediction'] = predictions
df['score'] = scores

df.to_csv('results.csv')  # Replace with desired file name.
