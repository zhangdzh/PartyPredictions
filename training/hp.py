from datasets import load_metric, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv', encoding='latin1', low_memory=False)
dropped_df = train_df.drop(labels=['bonica.rid',
                                   'congno',
                                   'NID',
                                   'source',
                                   'date',
                                   'nominate_dim2',
                                   'recipient.cfscore',
                                   'bioguide_id',
                                   'nominate_dim1',
                                   'bioname',
                                   'state_abbrev',
                                   'district_code',
                                   'icpsr',
                                   'congress',
                                   'chamber',
                                   'Unnamed: 0'], axis=1)

dropped_df = dropped_df.rename(columns={"party_code": "labels"})

dropped_df = dropped_df[dropped_df.labels != 328]
dropped_df.labels = dropped_df.labels//100 - 1  # 0 and 1
dropped_df.replace([np.inf, -np.inf], np.nan, inplace=True)
dropped_df = dropped_df.dropna()
print(dropped_df.info())


# sample df
# dropped_df = dropped_df.sample(100)

train_data = Dataset.from_pandas(dropped_df, preserve_index=False)

checkpoint = 'distilbert-base-uncased'  # used to be roberta-large
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length')


# DatasetDict
train_ds = train_data.train_test_split(test_size=0.5)  # 1/2 of set for test/val
test_valid = train_ds['test'].train_test_split(test_size=0.5)

# turn labels into ClassLabel
raw_train = train_ds['train']
new_features_train = raw_train.features.copy()
new_features_train["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_train = raw_train.cast(new_features_train)

raw_test = test_valid['test']
new_features_test = raw_test.features.copy()
new_features_test["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_test = raw_test.cast(new_features_test)

raw_valid = test_valid['train']
new_features_valid = raw_valid.features.copy()
new_features_valid["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_valid = raw_valid.cast(new_features_valid)

for ds in [raw_train, raw_test, raw_valid]:
    print(ds.features)

# tokenize each set separately then put into dictionary?
tokenized_train = raw_train.map(tokenize_function, batched=True)
tokenized_test = raw_test.map(tokenize_function, batched=True)
tokenized_valid = raw_valid.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = DatasetDict({
    'train': tokenized_train.remove_columns('text'),
    'test': tokenized_test.remove_columns('text'),
    'validation': tokenized_valid.remove_columns('text')
})

tokenized_datasets.set_format('torch')

# Trainer
training_args = TrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)  # evaluation strategy set

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'E:/zz2953/Dorothy/models/fine_tuned_distilbert', return_dict=True)


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")  # used to be glue, mnli
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Hyperparameter search starting now...")
trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10  # number of trials
)
