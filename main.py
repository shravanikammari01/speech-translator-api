from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

app = FastAPI()
model = whisper.load_model("base")

def roman_to_telugu(text: str) -> str:
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
    except:
        return text

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    result = model.transcribe(temp_audio_path)
    detected_text = result["text"].strip()
    detected_lang = result["language"]

    try:
        # Always produce Telugu output
        if detected_lang == "te":
            telugu_text = roman_to_telugu(detected_text)
        else:
            telugu_text = GoogleTranslator(source=detected_lang, target="te").translate(detected_text)

        # Always produce English output
        if detected_lang == "en":
            english_text = detected_text
        elif detected_lang == "te":
            english_text = GoogleTranslator(source="te", target="en").translate(telugu_text)
        else:
            english_text = GoogleTranslator(source=detected_lang, target="en").translate(detected_text)

    except Exception as e:
        english_text = "Translation failed"
        telugu_text = "Translation failed"

    os.remove(temp_audio_path)

    return {
        "detected_language": detected_lang,
        "english_text": english_text,
        "telugu_text": telugu_text
    }
