# app/app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os

# --- CRITICAL FIX: ABSOLUTE IMPORTS from src/ ---
from src.model_ops import load_model, train_model, predict_message
from src.db_ops import init_db, insert_message, fetch_training_data

# Ensure the models directory exists
MODELS_DIR = 'app/models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'a_very_secret_key_for_flash_messages' 

MODEL_PIPELINE = None
LAST_TRAINING_STATUS = "Model not yet trained."

init_db()

@app.before_request
def load_global_model():
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        MODEL_PIPELINE = load_model()

# --- Prediction Interface Route ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form['message']
        
        global MODEL_PIPELINE
        if MODEL_PIPELINE:
            label, probability = predict_message(input_text, MODEL_PIPELINE)
            prob_percent = f"{probability * 100:.2f}%"
            prediction_result = {
                'label': label.upper(), 
                'confidence': prob_percent,
                'is_spam': label == 'spam'
            }
        else:
            prediction_result = {'error': "ERROR: Model not trained. Need data and training."}

    return render_template('index.html', 
                           prediction=prediction_result, 
                           input_text=input_text,
                           training_url=url_for('training_manage')) 

# --- Dedicated Training Management Route ---

@app.route('/training-manage', methods=['GET'])
def training_manage():
    sample_count = fetch_training_data().shape[0]
    
    return render_template('training.html', 
                           train_status=LAST_TRAINING_STATUS,
                           sample_count=sample_count)


@app.route('/train', methods=['POST'])
def training_page():
    global MODEL_PIPELINE, LAST_TRAINING_STATUS
    
    flash("Training started. This may take a moment...", 'info')
    
    train_status = train_model()
    
    MODEL_PIPELINE = load_model()
    LAST_TRAINING_STATUS = train_status
    
    flash(train_status, 'success')
    
    return redirect(url_for('training_manage'))


@app.route('/feedback', methods=['POST'])
def feedback():
    """Handles user feedback on prediction accuracy."""
    message = request.form.get('message_text')
    correct_label = request.form.get('correct_label')
    
    if message and correct_label:
        insert_message(message, correct_label, priority=3) 
        flash(f"Feedback received! '{correct_label}' message added (Priority 3).", 'success')
    else:
        flash("Feedback failed.", 'danger')

    return redirect(url_for('index'))

# --- FUNCTION FOR CLEAN STARTUP ---
def run_initial_setup():
    """Executes initial training check and starts the Flask app."""
    global LAST_TRAINING_STATUS
    
    print("Running initial training check...")
    initial_status = train_model()
    
    # Update the global variable
    global LAST_TRAINING_STATUS 
    LAST_TRAINING_STATUS = initial_status
    
    print(f"Initial Status: {initial_status}")
    
    print("\n--- Starting Flask App (Navigate to http://127.0.0.1:5000/) ---")
    app.run(debug=True)

if __name__ == '__main__':
    run_initial_setup()