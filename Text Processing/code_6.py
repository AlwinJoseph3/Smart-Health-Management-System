import fitz  # PyMuPDF
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import json
import re
import tkinter as tk
from tkinter import filedialog

# Function to select PDF file using a file dialog
def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a PDF File",
        filetypes=[("PDF Files", "*.pdf")],
    )
    return file_path

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

# Preprocess extracted text
def preprocess_text(text):
    return " ".join(text.split())

# Chunk the text into smaller pieces for processing
def chunk_text(text, max_tokens=450):
    words = text.split()
    chunks = []
    while words:
        chunk = " ".join(words[:max_tokens])
        chunks.append(chunk)
        words = words[max_tokens:]
    return chunks

# Load model and tokenizer
def load_model():
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# Summarize text
def summarize_text(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,  
        num_beams=8,  
        length_penalty=0.8,  
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Extract key information using AI-based pattern matching
def extract_key_information(summary):
    info = {"Name": "Not Available", "Age": "Not Available", "Gender": "Not Available", "Diagnosis": "Not Available", "Treatment": "Not Available"}

    # Extract Name: The first two words could be the patient's name
    name_match = re.search(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s", summary)
    if name_match:
        info["Name"] = name_match.group(1)

    # Extract Age: Looks for patterns like "72 year old"
    age_match = re.search(r"(\d{1,3})\s?(?:year old|years old|y/o|yr old)", summary, re.IGNORECASE)
    if age_match:
        info["Age"] = age_match.group(1)

    # Extract Gender: Search for "Male" or "Female" near the beginning
    gender_match = re.search(r"\b(Male|Female|Other)\b", summary, re.IGNORECASE)
    if gender_match:
        info["Gender"] = gender_match.group(1)

    # Extract Diagnosis: Find conditions mentioned
    diagnosis_match = re.search(r"history of (.*?)[,.]", summary, re.IGNORECASE)
    if diagnosis_match:
        info["Diagnosis"] = diagnosis_match.group(1).strip()

    # Extract Treatment: Find mentions of medications
    treatment_match = re.search(r"treated with (.*?)[,.]", summary, re.IGNORECASE)
    if treatment_match:
        info["Treatment"] = treatment_match.group(1).strip()

    # Add the complete summary text
    info["Summary"] = summary

    return info

# Main function to run the process
if __name__ == "__main__":
    # Step 1: Allow the user to select a PDF file
    pdf_path = select_pdf_file()

    if not pdf_path:
        print("No PDF file selected. Exiting...")
    else:
        print(f"Processing file: {pdf_path}")

        # Step 2: Extract text from the selected PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Step 3: Preprocess the extracted text
        preprocessed_text = preprocess_text(extracted_text)

        # Step 4: Chunk the text into smaller pieces
        chunks = chunk_text(preprocessed_text, max_tokens=450)

        # Step 5: Load the Pegasus model and tokenizer
        tokenizer, model, device = load_model()

        # Step 6: Generate summaries for each chunk
        summaries = [summarize_text(chunk, tokenizer, model, device) for chunk in chunks]

        # Step 7: Combine the chunk summaries into a final summary
        final_summary = "\n".join(summaries)

        # Step 8: Extract key information for JSON output
        json_output = extract_key_information(final_summary)

        # Step 9: Save the JSON output to a file
        with open("summary.json", "w", encoding="utf-8") as json_file:
            json.dump(json_output, json_file, indent=4)

        # Print the JSON output to console
        print(json.dumps(json_output, indent=4))

        # Confirmation
        print("Summary generated and saved to 'summary.json'.")
