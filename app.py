
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------- ASR ----------
asr_khowar = pipeline(
    "automatic-speech-recognition",
    model="huggingface.com/aizazayyubi/models/khowar-asrl"
)

asr_pashto = pipeline(
    "automatic-speech-recognition",
    model="huggingface.com/aizazayuubi/models/pashto-asr"
)

# ---------- Language ID (optional placeholder) ----------
def detect_language(audio_path):
    return "khowar"

# ---------- Hate Speech Classifier ----------
tokenizer = AutoTokenizer.from_pretrained("your-org/hate-speech-khowar-pashto")
model = AutoModelForSequenceClassification.from_pretrained(
    "your-org/hate-speech-khowar-pashto"
)

hate_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# ---------- Full Pipeline ----------
def process_audio(audio_path):
    lang = detect_language(audio_path)

    if lang == "khowar":
        transcription = asr_khowar(audio_path)["text"]
    else:
        transcription = asr_pashto(audio_path)["text"]

    prediction = hate_classifier(transcription)

    return {
        "language": lang,
        "transcription": transcription,
        "prediction": prediction
    }

# ---------- Example ----------
if __name__ == "__main__":
    result = process_audio("sample.wav")
    print(result)

