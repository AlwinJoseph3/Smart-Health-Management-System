import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to prepare data
def prepare_data():
    # Load the CSV file containing your discharge summaries
    df = pd.read_csv('D:/1_MAIN PROJECT/Text Summarization/Model/discharge_summaries.csv')

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2)

    # Tokenizer to tokenize our inputs and outputs
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')

    def tokenize_function(examples):
        # Tokenizing the input and output (summary)
        inputs = tokenizer(examples['report'], max_length=512, truncation=True, padding='max_length')
        outputs = tokenizer(examples['summary'], max_length=128, truncation=True, padding='max_length')
        inputs['labels'] = outputs['input_ids']
        return inputs

    # Convert to a dataset
    train_dataset = train_df[['report', 'summary']].to_dict(orient='records')
    val_dataset = val_df[['report', 'summary']].to_dict(orient='records')

    # Tokenize datasets
    train_dataset = list(map(tokenize_function, train_dataset))
    val_dataset = list(map(tokenize_function, val_dataset))

    return train_dataset, val_dataset, tokenizer

# Main function to start training
def main():
    # Check if GPU is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_dataset, val_dataset, tokenizer = prepare_data()

    # Load model and move it to the GPU if available
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,               
    weight_decay=0.01,              
    logging_dir='./logs',            
    evaluation_strategy="epoch",    # Make sure this matches save_strategy
    save_strategy="epoch",          # Set to 'epoch' for both save and eval strategies
    load_best_model_at_end=True     # This will load the best model at the end of training
)

    # Trainer setup
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,             
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
