from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder for PDFs
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure CORS for local frontend and backend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000", "http://localhost:5000"]
    }
})

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------ Helper Functions ------------------

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


# def generate_health_response_with_history(current_message, chat_history=None):
#     """Generates a health-related AI response using Gemini with chat history context"""
    
#     # Build context from chat history
#     context = ""
#     if chat_history and len(chat_history) > 0:
#         context = "Previous conversation context:\n"
#         for chat in chat_history[-10:]:  # Use last 10 messages for context
#             if chat.get('message'):
#                 context += f"User: {chat['message']}\n"
#             if chat.get('response'):
#                 context += f"Assistant: {chat['response']}\n"
#         context += "\n"
    
#     prompt = f"""
#     You are an empathetic AI Health Assistant. 
    
#     {context}
    
#     Current user message: "{current_message}"
    
#     Based on the current message and conversation history, provide:
#     1. A short and crisp explanation of what could be happening (in layman's terms).
#     2. Possible causes of the symptoms.
#     3. Recommended next steps or precautions and medications.
    
#     Keep your response helpful, empathetic, and medically responsible. Always recommend consulting healthcare professionals for serious concerns.
#     """

#     try:
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print("AI generation error:", e)
#         return "Sorry, I couldn't process your request. Please try again later."
def generate_health_response_with_history(current_message, chat_history=None):
    context = ""
    if chat_history:
        for chat in chat_history[-5:]:
            if chat.get('message'):
                context += f"User: {chat['message']}\n"
            if chat.get('response'):
                context += f"Assistant: {chat['response']}\n"

    prompt = f"""
    You are an empathetic AI Health Assistant.

    Previous context:
    {context}

    Current user message: "{current_message}"

    Give a short, clear health explanation in 3 parts:
    1. What may be happening (simple language)
    2. Possible causes
    3. Precautions and medications intake and when to see a doctor
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt, stream=True)
        final_text = ""
        for chunk in response:
            if chunk.text:
                final_text += chunk.text
        return final_text.strip()
    except Exception as e:
        print("AI generation error:", e)
        return "Sorry, I couldn't process your request. Please try again later."



def generate_health_response_from_file(file_text, chat_history=None):
    """Generates health response from uploaded file content with context"""
    
    # Build context from chat history
    context = ""
    if chat_history and len(chat_history) > 0:
        context = "Previous conversation context:\n"
        for chat in chat_history[-5:]:  # Use last 5 messages for context
            if chat.get('message'):
                context += f"User: {chat['message']}\n"
            if chat.get('response'):
                context += f"Assistant: {chat['response']}\n"
        context += "\n"
    
    prompt = f"""
    You are an empathetic AI Health Assistant analyzing uploaded medical documents.
    
    {context}
    
    Document content: "{file_text}"
    
    Based on the document content and conversation history, provide:
    1. Summary of the key health information from the document
    2. Important findings or concerns
    3. Recommendations for next steps
    4. Questions to discuss with healthcare providers
    
    Keep your response helpful and medically responsible.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("AI generation error:", e)
        return "Sorry, I couldn't analyze the document. Please try again later."

# ------------------ Routes ------------------

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with optional history context"""
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        chat_history = data.get("history", [])  # Optional chat history
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        start_time = time.time()
        
        # Generate AI response with history context
        response = generate_health_response_with_history(user_message, chat_history)
        
        processing_time = int((time.time() - start_time) * 1000)  # in milliseconds
        
        return jsonify({
            "response": response,
            "processing_time": processing_time,
            "success": True
        })

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    """Handle file upload and analysis with optional history context"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Get chat history from form data if provided
        chat_history = []
        if 'history' in request.form:
            import json
            try:
                chat_history = json.loads(request.form['history'])
            except:
                chat_history = []

        # Save uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        start_time = time.time()

        try:
            # Extract text from PDF
            if file.filename.lower().endswith('.pdf'):
                extracted_text = extract_text_from_pdf(file_path)
                if not extracted_text:
                    return jsonify({"error": "Could not extract text from PDF"}), 400
                
                # Generate AI response
                response = generate_health_response_from_file(extracted_text, chat_history)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                return jsonify({
                    "response": response,
                    "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                    "processing_time": processing_time,
                    "file_info": {
                        "filename": file.filename,
                        "size": os.path.getsize(file_path)
                    },
                    "success": True
                })
            else:
                return jsonify({"error": "Only PDF files are supported currently"}), 400
                
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        print("File analysis error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "OK", 
        "message": "AI Health Assistant Flask API is running!",
        "version": "2.0"
    })


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "AI Health Assistant Flask API", 
        "endpoints": {
            "/chat": "POST - Send chat message with optional history",
            "/analyze-file": "POST - Upload and analyze medical documents",
            "/health": "GET - Health check"
        }
    })


# Run app
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to avoid conflicts
