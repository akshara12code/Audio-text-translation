from flask import Flask, request, jsonify, render_template
import os
import whisper
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

# Create static folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load Whisper model
print("Loading Whisper model...")
try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded!")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    exit(1)

# Load Gen AI Translation Model
print("Loading Gen AI translation model...")
model_name = "facebook/m2m100_418M"

try:
    print("📥 Loading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(
        model_name, 
        resume_download=True,
        local_files_only=False
    )
    print("✅ Tokenizer loaded!")
    
    print("📥 Loading translation model (this may take a while if downloading)...")
    print("⏳ Please wait... Do NOT interrupt this process!")
    
    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name,
        resume_download=True,
        local_files_only=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True  # Use less memory during loading
    )
    print("✅ Gen AI translation model loaded successfully!")
    print("🎉 All models ready! Starting Flask server...")
    
except Exception as e:
    print(f"❌ Error loading translation model: {e}")
    print("💡 Suggestion: Check your internet connection and try again")
    print("💡 The download may have been interrupted. It will resume automatically.")
    exit(1)

# Language mapping for M2M100
LANGUAGE_CODES = {
    'Spanish': 'es',
    'French': 'fr', 
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Gujarati': 'gu'
}

def translate_with_gen_ai(text, target_lang_code):
    """Use Gen AI model for translation"""
    try:
        print(f"🔄 Translating to {target_lang_code}...")
        
        # Set source language to English
        tokenizer.src_lang = "en"
        
        # Encode the text
        encoded = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        # Generate translation using the Gen AI model
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded, 
                forced_bos_token_id=tokenizer.get_lang_id(target_lang_code),
                max_length=512,
                num_beams=5,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print("✅ Translation completed!")
        return translation
        
    except Exception as e:
        print(f"❌ Gen AI translation error: {e}")
        return f"Translation failed: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            print("📨 Received upload request")
            
            # Get form data
            language = request.form.get('language')
            file = request.files.get('file')
            
            # Basic validation
            if not language:
                return jsonify({"error": "No language selected"}), 400
            
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            
            if not file.filename:
                return jsonify({"error": "No file selected"}), 400
            
            # Save uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"💾 File saved: {file_path}")
            print("🎤 Starting Whisper transcription...")
            
            # Use Whisper for transcription
            result = whisper_model.transcribe(file_path)
            transcript_text = result["text"]
            detected_language = result.get("language", "unknown")
            
            print(f"✅ Transcription complete!")
            print(f"📝 Detected language: {detected_language}")
            print(f"📄 Transcript preview: {transcript_text[:100]}...")
            
            # Validate target language
            if language not in LANGUAGE_CODES:
                return jsonify({"error": f"Unsupported language: {language}"}), 400
            
            # Use Gen AI for translation
            target_lang_code = LANGUAGE_CODES[language]
            translation = translate_with_gen_ai(transcript_text, target_lang_code)
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
                print(f"🗑️ Cleaned up: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Could not clean up file: {cleanup_error}")
            
            print("🎉 Process completed successfully!")
            
            return jsonify({
                "transcript": transcript_text,
                "translation": translation,
                "detected_language": detected_language,
                "target_language": language,
                "model_info": "Using Facebook M2M100 Gen AI for translation"
            })
            
        except Exception as e:
            print(f"❌ Error in processing: {str(e)}")
            # Clean up file if it exists
            try:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            return jsonify({"error": f"Server error: {str(e)}"}), 500
    
    # For GET request, render the template
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VOICE TRANSLATION APP")
    print("=" * 60)
    print("📂 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("🌍 Supported languages:", len(LANGUAGE_CODES))
    print("🤖 Models loaded successfully!")
    print("🌐 Starting server...")
    print("=" * 60)
    
    # Run without debug mode to prevent auto-reloading during model download
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)