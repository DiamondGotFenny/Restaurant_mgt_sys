from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import azure.cognitiveservices.speech as speechsdk
import base64
import time

_ = load_dotenv(find_dotenv())

#creadentials for openai
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
model3_name=os.getenv("OPENAI_MODEL_3")
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

#configure for system output
speech_config.speech_synthesis_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer=speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

 # tts sentence end mark
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app=FastAPI(middleware=middleware)



client_address = os.getenv("CLIENT_ADDRESS")

# In-memory chat history
chat_history = [ {"role": "system", "content": "Assistant's name is Jenna. She is a virtual receptionist for a restaurant. She can help user with user's needs.But Jenna will only answer the question based on restaurant information, if user's question is not related to the restaurant information, or content not included in the restaurant information, she will not be able to answer it. Jenna's answer should be short and clean,like spoken conversation. REMEMBER THAT!"},
                {"role": "system", "content": "the restaurant information:restaurant name is Kiwi's Day. Today's special is chicken curry and beef curry. We have 5 tables available for 2 people and 3 tables available for 4 people.now we have 2 tables for 2 people, 1 table for 4 people available. our business hours are 11:00 am to 10:00 pm."},
                {"role": "assistant", "content": "Hi this is Jenna, nice talke to you and how may I help",},]


class Chat_Request(BaseModel):
    message: str

class Voice_Request(BaseModel):
    data: UploadFile = File(...)

def is_audio_file(file: UploadFile) -> bool:
    """
    Check if the file is a wav audio file based on its MIME type.
    """
    accepted_mime_types = "audio/wav"
    return file.content_type == accepted_mime_types
    
# Define the speech-to-text function
async def speech_to_text(audio_file: UploadFile):
    # Check if the file is a wav file
    if not is_audio_file(audio_file):
        return {"error": "The file is not valid wav file."}

    try:
        # Save the audio file
        with open('recording.wav', 'wb') as f:
            f.write(await audio_file.read())

        # Create an AudioConfig object
        audio_input = speechsdk.AudioConfig(filename='recording.wav')
        

        # Initialize SpeechRecognizer
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    except Exception as e:
        print(f"Error: {e}")
        return 'sorry I did not catch you, could please repeat it?'

    speech_recognition_result = speech_recognizer.recognize_once()

    # Handling different speech recognition outcomes
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return 'sorry I did not catch you, could please repeat it?'
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            return None
        return None

# Define the text-to-speech function
def text_to_speech(text):
    try:
        result = speech_synthesizer.start_speaking_text_async(text).get()
        audio_data_stream = speechsdk.AudioDataStream(result)
        audio_buffer = bytes(16000)
        filled_size = audio_data_stream.read_data(audio_buffer)
        while filled_size > 0:
            yield audio_buffer[:filled_size]
            filled_size = audio_data_stream.read_data(audio_buffer)
    except Exception as ex:
        print(f"Error synthesizing audio: {ex}")
        yield b''
    
# get the response from LLM function
def response_from_LLM(input_text):
      #check if the input is valid string
    if not input_text:
        return {"error": "The input is not a valid string."}
    try:
        chat_history.append({"role": "user", "content": input_text,})
        response = client.chat.completions.create(
            model=model3_name, # model = "deployment_name".
            messages=chat_history,
        )
        gpt_response=response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": gpt_response,})
        print(chat_history)
        return {"response": gpt_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


  
@app.post("/chat-text/")
async def chat_text(chat: Chat_Request):
   response_from_LLM(chat.message)

#define speech to text chat route
@app.post("/chat-speech-to-text/")
async def chat_speech_to_text(data: UploadFile = File(...)):
  # Check if the file is an wav audio file
    if not is_audio_file(data):
        return {"error": "The file is not an audio file."}
    
      # Assuming `speech_to_text` function can handle audio file
    input_text = await speech_to_text(data)

    # Check if speech to text was successful
    if input_text is None:
        return {"error": "Speech to text conversion failed"}

    if input_text:
        return {'input_text':input_text}
    else:
        return {"error": "Unable to recognize the audio"}
#todo: push the response text to the history
# Define text to speech chat route
@app.post("/chat-text-to-speech/")
async def chat_text_to_speech(data: Chat_Request):
    # Send the text to the LLM and get the response
    response = response_from_LLM(data.message)

    # Check if the response is successful
    if response:
        # Convert the response to speech
        audio_stream = text_to_speech(response["response"])

        if  audio_stream:
            return StreamingResponse(audio_stream, media_type='audio/wav')
        else:
            return {"error": "Unable to synthesize audio"}
    else:
        return {"error": "Unable to get response from LLM"}

# Define a route to get the chat history
@app.get("/chat_history/")
async def get_chat_history():
     # Filter out system messages
    filtered_chat_history = [message for message in chat_history if message["role"] != "system"]
    return {"chat_history": filtered_chat_history}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)