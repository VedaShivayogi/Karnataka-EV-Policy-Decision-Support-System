"""
Module 10 - Voice Assistant
Flow: microphone -> Whisper (speech-to-text) -> LLM -> Piper/Coqui (text-to-speech) -> speaker

This module can also be used purely as text-in/audio-out (skip Whisper) if you
just want to test: "How will fuel prices affect EV adoption?"

Run (voice mode, needs a microphone):
    python modules/voice_assistant.py --voice

Run (text mode, no microphone needed):
    python modules/voice_assistant.py --text "How will fuel prices affect EV adoption?"
"""

import sys
import os
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import chat

SYSTEM_PROMPT = (
    "You are an EV policy voice assistant for Karnataka's transport department. "
    "Answer in 3-5 short, spoken-friendly sentences (no markdown, no bullet points, "
    "since this will be read aloud by a TTS engine)."
)


def record_and_transcribe(duration_seconds: int = 6) -> str:
    """Record from mic and transcribe with OpenAI Whisper (runs locally, free)."""
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    import whisper

    print(f"Recording for {duration_seconds}s... speak now.")
    fs = 16000
    audio = sd.rec(int(duration_seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    tmp_wav = os.path.join(config.OUTPUT_DIR, "mic_input.wav")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    wavfile.write(tmp_wav, fs, audio)

    print(f"Transcribing with Whisper ({config.WHISPER_MODEL_SIZE})...")
    model = whisper.load_model(config.WHISPER_MODEL_SIZE)
    result = model.transcribe(tmp_wav)
    text = result["text"].strip()
    print(f"You said: {text}")
    return text


def answer_question(question: str) -> str:
    return chat(question, system=SYSTEM_PROMPT)


def speak_piper(text: str, out_wav: str):
    """Text-to-speech using Piper (free, fast, runs locally)."""
    voice = config.PIPER_VOICE_MODEL
    cmd = ["piper", "--model", voice, "--output_file", out_wav]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def speak_coqui(text: str, out_wav: str):
    """Text-to-speech using Coqui TTS (free, higher quality, heavier)."""
    from TTS.api import TTS

    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=text, file_path=out_wav)


def play_audio(path: str):
    try:
        import simpleaudio as sa

        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play().wait_done()
    except Exception:
        # Fallback to OS player
        if sys.platform == "darwin":
            subprocess.run(["afplay", path])
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", path])
        else:
            os.startfile(path)  # Windows


def run(voice_mode: bool, text_input: str):
    if voice_mode:
        question = record_and_transcribe()
    else:
        question = text_input or "How will fuel prices affect EV adoption?"
        print(f"Question: {question}")

    answer = answer_question(question)
    print(f"\nAssistant: {answer}\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_wav = os.path.join(config.OUTPUT_DIR, "assistant_reply.wav")

    try:
        if config.TTS_ENGINE == "coqui":
            speak_coqui(answer, out_wav)
        else:
            speak_piper(answer, out_wav)
        print(f"Saved speech to {out_wav}")
        play_audio(out_wav)
    except Exception as e:
        print(f"[warn] TTS playback skipped ({e}). Text answer is above.")

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="store_true", help="Use microphone input")
    parser.add_argument("--text", type=str, default="", help="Ask a question via text instead")
    args = parser.parse_args()
    run(args.voice, args.text)
