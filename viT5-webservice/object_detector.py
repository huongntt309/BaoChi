from flask import request, Response, Flask, render_template
from waitress import serve
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
import os

app = Flask(__name__)

model_predictions = None
tokenizer = None
# set up functions 
def setup():
    # Load the model_predictions
    global model_predictions
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "checkpoint-2740")
    model_predictions = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Load tokenizer
    global tokenizer
    MY_HUGGINGFACE_TOKEN = "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"
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", token=MY_HUGGINGFACE_TOKEN)
    
    print("Set up model and tokenizer successfully")
    print("Open local link: http://localhost:8080/")


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    index_html_path = os.path.join(current_dir, "index.html")
    with open(index_html_path, encoding="utf-8") as file:
        html_content = file.read()
    return html_content


@app.route("/predict", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Input: corpus
        Return: type of corpus
    """
    # Load the corpus from the request
    data = request.json
    corpus = data.get('corpus', '')  # Extracting the 'corpus' field from the JSON data
    print("corpus:" , corpus)
    global model_predictions
    # Check if model_predictions object is initialized
    if model_predictions is None:
        return Response(json.dumps({"error": "Model not initialized"}), mimetype='application/json')

    # Perform detection
    max_target_length = 256
    inputs = tokenizer(corpus, return_tensors="pt")
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
    print("predicted_text:" , predicted_text)
    
    result = {"classification": predicted_text}

    return Response(json.dumps(result), mimetype='application/json')


# Call the setup function before starting the server
setup()

serve(app, host='0.0.0.0', port=8080)