# import 
import os
import gradio as gr
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 


# set up functions 
MY_HUGGINGFACE_TOKEN = "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"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", token=MY_HUGGINGFACE_TOKEN)

# Load model
model_path = "E:\\code_project\\UET\\flask\\src\\viT5-gradio\\checkpoint-2740"
model_predictions = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Model loaded successfully")
def predict_type(text):
    global tokenizer
    global model_predictions
    
    # preprocess the text
    max_target_length = 256
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # model predictions
    output = model_predictions.generate(
        input_ids=input_ids,
        max_length=max_target_length,
        attention_mask=attention_mask,
    )
    
    # decode output
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return predicted_text

# Gradio interface
gr.Interface(fn=predict_type, inputs="text", outputs="text").launch(share=True)