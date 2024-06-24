from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import azure.cognitiveservices.speech as speechsdk
import wave
import io
import struct

_ = load_dotenv(find_dotenv())

#creadentials for openai
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
api_version="2024-02-15-preview"
model3_name=os.getenv("OPENAI_MODEL_3")
model4_name=os.getenv("OPENAI_MODEL_4_16")


client=AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
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
 # Creates a speech synthesizer with a null output stream.
    # This means the audio output data will not be written to any output channel.
    # You can just get the audio from the result.
speech_synthesizer=speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

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
init_chat = [{"role": "system", "content": "Assistant's name is Jenna. She is a virtual receptionist for a restaurant. She can help user with user's needs.Jenna need to ask user's name, how many people will join the party, what time they will be arrived if user want to book a table. if Jenna don't know any of those three infomation, she will have to ask the user untill she have all three info. Jenna will only answer the question based on restaurant information, if user's question is not related to the restaurant information, or content not included in the restaurant information, she will not be able to answer it. Jenna's answer should be short and clean,like spoken conversation. REMEMBER THAT!"},
                {"role": "system", "content": "the restaurant information:restaurant name is Kiwi's Day. Today's special is chicken curry and beef curry. We have 5 tables available for 2 people and 3 tables available for 4 people.now we have 2 tables for 2 people, 1 table for 4 people available. our business hours are 11:00 am to 10:00 pm."},
                {"role": "assistant", "content": "Hi this is Jenna, nice talke to you and how may I help",},
                ]
# init In-memory chat history
chat_history = [{"role": "system", "content": "Assistant's name is Jenna. She is a virtual receptionist for a restaurant. She can help user with user's needs.Jenna need to ask user's name, how many people will join the party, what time they will be arrived if user want to book a table. if Jenna don't know any of those three infomation, she will have to ask the user untill she have all three info. Jenna will only answer the question based on restaurant information, if user's question is not related to the restaurant information, or content not included in the restaurant information, she will not be able to answer it. Jenna's answer should be short and clean,like spoken conversation. REMEMBER THAT!"},
                {"role": "system", "content": "the restaurant information:restaurant name is Kiwi's Day. Today's special is chicken curry and beef curry. We have 5 tables available for 2 people and 3 tables available for 4 people.now we have 2 tables for 2 people, 1 table for 4 people available. our business hours are 11:00 am to 10:00 pm."},
                {"role": "assistant", "content": "Hi this is Jenna, nice talke to you and how may I help",},
                ]


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
        print("The file is not valid wav file.")
        return 'invalid wav file'

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
        return None

    speech_recognition_result = speech_recognizer.recognize_once()

    # Handling different speech recognition outcomes
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return None
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Cancellation Error details: {}".format(cancellation_details.error_details))
            return None
        return None
    
    
def create_wav_header(channels, sample_rate, bits_per_sample):
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample, b'data', 0)
    print('header setted')
    return header

# Define the text-to-speech stream function
async def text_to_speech_stream(text):
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data_stream = speechsdk.AudioDataStream(result)
            audio_data_stream.position = 0
           
            # Yield WAV header
            yield create_wav_header(1, 16000, 16)

            # Stream audio data in chunks
            chunk_size = 16000  # You can adjust this value
            audio_buffer = bytes(chunk_size)
            while True:
                filled_size = audio_data_stream.read_data(audio_buffer)
                if filled_size > 0:
                    print(f"{filled_size} bytes received.")
                    yield audio_buffer[:filled_size]
                else:
                    break

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
    except Exception as ex:
        print(f"Error synthesizing audio: {ex}")
        yield b''

# Define the text-to-speech function
async def text_to_speech_completed(text):
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(text))
            return result
        elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
    except Exception as ex:
        print(f"Error synthesizing audio: {ex}")
        return None          
    
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
   return response_from_LLM(chat.message)

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
        return {"error": "your speech is not recognized, could you please repeat it?"}
    # if the input_text string equal to 'invalid wav file' string, we return error
    if input_text == 'invalid wav file':
        return {"error":'your file is not a valid wav file'}
    if input_text:
        return {'input_text':input_text}
    else:
        return {"error": "Unable to recognize the audio"}
    
# Define text to speech chat route
@app.post("/chat-text-to-speech/")
async def chat_text_to_speech(data: Chat_Request):
    # Send the text to the LLM and get the response
    response = response_from_LLM(data.message)
    # Check if the response is successful
    if response:
        # Convert the response to speech
        audio_stream = text_to_speech_stream(response["response"])
        
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
    
    #create a route that can clean the chat history
@app.post("/clear_chat_history/")
async def delete_chat_history():
    global chat_history
    chat_history=init_chat.copy()
    return {"message": "Chat history has been cleared."}