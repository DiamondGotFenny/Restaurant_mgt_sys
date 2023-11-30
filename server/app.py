from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import azure.cognitiveservices.speech as speechsdk
import base64

_ = load_dotenv(find_dotenv())

#creadentials for openai
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
model_name=os.getenv("OPENAI_MODEL_3")
model4_name=os.getenv("OPENAI_MODEL_4_16")

client=AzureOpenAI(
    api_key=api_key,
    api_version="2023-05-15",
    azure_endpoint=api_base
)

#creadentials for azure speech
speech_key=os.getenv("AZURE_SPEECH_KEY")
service_region =os.getenv("AZURE_SPEECH_REGION")
speech_endpoint =os.getenv("AZURE_SPEECH_ENDPOINT")

# Set up Azure speech
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

#configure for user input 
speech_config.speech_recognition_language="en-US"
#audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
#speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#configure for system output
speech_config.speech_synthesis_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer=speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

 # tts sentence end mark
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]



app=FastAPI()

client_address = os.getenv("CLIENT_ADDRESS")

# Add CORS middleware to allow requests from your client's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[client_address],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory chat history
chat_history = [ {"role": "system", "content": "Assistant's name is Janna.She is a virtual receptionist for a restaurant. She can help you with your needs."},
                {"role": "system", "content": "today's special is chicken curry and beef curry. we have 5 tables available for 2 people and 3 tables available for 4 people.now we have 2 tables for 2 people, 1 table for 4 people available. our business hours are 11:00 am to 10:00 pm."},
                {"role": "assistant", "content": "Hi this is Janna, nice talke to you and how may I help",},]

greeting=f'Hi this is Janna, nice talke to you and how may I help?'

class Chat_Request(BaseModel):
    message: str

    
# Define the speech-to-text function
def speech_to_text(audio_file: UploadFile):
    print("Speak into your microphone.")
    # Create an AudioConfig object with the path to your audio/wav file
    audio_input = speechsdk.AudioConfig(stream=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    speech_recognition_result = speech_recognizer.recognize_once()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return ""
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
            return ""
        return ""

# Define the text-to-speech function
def text_to_speech(text):
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Text-to-speech conversion successful.")
            # Encode the audio data into Base64
            encoded_audio = base64.b64encode(result.audio_data)
            return encoded_audio
        else:
            print(f"Error synthesizing audio: {result}")
            return False
    except Exception as ex:
        print(f"Error synthesizing audio: {ex}")
        return False
    
# get the response from LLM function
def response_from_LLM(input_text):
    try:
        chat_history.append({"role": "user", "content": input_text,})
        response = client.chat.completions.create(
            model=model_name, # model = "deployment_name".
            messages=chat_history,
        )
        gpt_response=response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": gpt_response,})
        print(chat_history)
        return {"response": gpt_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def is_audio_file(file: UploadFile) -> bool:
    """
    Check if the file is an audio file based on its MIME type.
    """
    # List of accepted audio MIME types
    accepted_mime_types = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/mp3"]
    return file.content_type in accepted_mime_types
  
@app.post("/chat-text/")
async def chat_text(chat: Chat_Request):
    try:
        chat_history.append({"role": "user", "content": chat.message,})
        response = client.chat.completions.create(
            model=model_name, # model = "deployment_name".
            messages=chat_history
        )
        gpt_response=response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": gpt_response,})
        print(chat_history)
        return {"response": gpt_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#define voice chat route
@app.post("/chat-voice/")
async def chat_voice(chat: Chat_Request):
    print(chat.message)

    # Send the text to the LLM and get the response
    response = response_from_LLM(chat.message)
    response_text=response['response']
    # Convert the response to speech
    encoded_audio = text_to_speech(response_text)

    if encoded_audio:
        # return {"response": response["response"], "audio": encoded_audio}
        return {"text_data": response_text, "audio_data": encoded_audio.decode("utf-8")}
    else:
        return {"error": "Unable to synthesize audio"}


# Define a route to get the chat history
@app.get("/chat_history/")
async def get_chat_history():
     # Filter out system messages
    filtered_chat_history = [message for message in chat_history if message["role"] != "system"]
    return {"chat_history": filtered_chat_history}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)