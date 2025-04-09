import gradio as gr
import os
import re
import requests
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# === Nutanix API settings ===
NUTANIX_API_URL = os.getenv("NUTANIX_API_URL")  # Set this in your environment!
NUTANIX_API_KEY = os.getenv("NUTANIX_API_KEY")  # Set this in your environment!
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript_from_url(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, "❌ Invalid YouTube URL."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text, "✅ Transcript fetched successfully."
    except TranscriptsDisabled:
        return None, "🚫 Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "❌ No transcript available."
    except Exception as e:
        return None, f"⚠️ Error: {str(e)}"

def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"⚠️ PDF read error: {str(e)}"

def nutanix_chat(messages):
    if not NUTANIX_API_KEY:
        return "❌ Nutanix API key not set."

    headers = {
        "Authorization": f"Bearer {NUTANIX_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "nim-llama-3-1-8b",  # Adjust to your available models
        "messages": messages,
        "temperature": 0.7
    }

    try:
        res = requests.post(NUTANIX_API_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Nutanix GPT error: {str(e)}"

# Session state: stores combined context
session_context = {"transcript": "", "pdf": "", "chat_history": []}

def handle_upload(url, pdf_file):
    transcript, status = get_transcript_from_url(url)
    if not transcript:
        return status, "", "", []

    pdf_text = read_pdf(pdf_file)
    if pdf_text.startswith("⚠️"):
        return pdf_text, transcript, "", []

    # Save for chat session
    session_context["transcript"] = transcript
    session_context["pdf"] = pdf_text
    session_context["chat_history"] = []

    comparison_prompt = (
        "Compare the following YouTube transcript and PDF content. "
        "Summarize their similarities and differences.\n\n"
        f"Transcript:\n{transcript[:3000]}\n\nPDF:\n{pdf_text[:3000]}"
    )

    response = nutanix_chat([{"role": "user", "content": comparison_prompt}])
    return "✅ Files processed. You can now chat.", transcript, response, []

def chat_with_context(user_message, chat_history):
    base_context = (
        "You are an assistant who answers questions based on a YouTube video transcript and a PDF document. "
        "Use the following information as reference.\n\n"
        f"Transcript:\n{session_context['transcript'][:3000]}\n\n"
        f"PDF:\n{session_context['pdf'][:3000]}"
    )

    messages = [
        {"role": "system", "content": base_context}
    ]

    # Build chat history
    for pair in chat_history:
        messages.append({"role": "user", "content": pair[0]})
        messages.append({"role": "assistant", "content": pair[1]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    reply = nutanix_chat(messages)

    # Save to session
    session_context["chat_history"].append((user_message, reply))

    return session_context["chat_history"]

# === Gradio UI ===
with gr.Blocks() as app:
    gr.Markdown("## 🎥📄 YouTube + PDF Chat App (Powered by Nutanix AI)")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

    upload_button = gr.Button("Process Inputs")

    status_output = gr.Textbox(label="Status")
    transcript_output = gr.Textbox(label="Transcript", lines=8)
    comparison_output = gr.Textbox(label="Comparison Summary", lines=8)
    
    gr.Markdown("### 💬 Ask Questions About the Video + PDF")

    chatbot = gr.Chatbot()
    message_input = gr.Textbox(label="Ask a question")

    message_input.submit(
        fn=chat_with_context,
        inputs=[message_input, chatbot],
        outputs=chatbot
    )

    upload_button.click(
        fn=handle_upload,
        inputs=[url_input, pdf_input],
        outputs=[status_output, transcript_output, comparison_output, chatbot]
    )

app.launch()

