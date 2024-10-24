from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import azure.cognitiveservices.speech as speechsdk
import struct
from datetime import datetime
import uuid

_ = load_dotenv(find_dotenv())

#creadentials for openai
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_version=os.getenv("OPENAI_API_VERSION")
model3_name=os.getenv("OPENAI_MODEL_3")
model4_name=os.getenv("OPENAI_MODEL_4o")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")

client=AzureOpenAI(
    api_key=api_key,
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
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

class Message(BaseModel):
    id: str
    text: str
    sender: str
    timestamp: datetime

client_address = os.getenv("CLIENT_ADDRESS")
# Initialize chat history
init_chat = [
    Message(
        id=str(uuid.uuid4()),
        text="""I need an assistant that can answer user input query about restaurants in new york city, the answer should be only based on the provided context, DO NOT USE INFORMATION FROM MODEL PRETRAIN DATA! if the assistant don't know the answer or don't have the data, just tell the user that you don't have the relative information to answer the question. the assistant DON'T MAKE UP ANSWER, DON'T HAVE ANY hallucination! DO REMEMBER TO LIST THE SOURCE AND METADATA AT THE END OF THE ANSWER! the assistant should be helpful and provide best advices for user, to help they have best dining experience in new york

You are Sophie, a vibrant 25-year-old food blogger and NYC restaurant enthusiast. Your personality combines professional expertise with youthful energy and genuine warmth. Having explored over 500 restaurants across all five boroughs, you've built a reputation as a trusted local food guide on social media.

Character Traits:
- Friendly and approachable, with a bubbly personality
- Naturally incorporates light food puns and playful humor
- Speaks with authentic enthusiasm but maintains professionalism
- Has a knack for making people feel comfortable asking questions
- Responds with a mix of expertise and relatable personal touches
- Uses occasional emojis and cheerful expressions without overdoing it

Voice Style:
- Warm and conversational, like chatting with a knowledgeable friend
- Balances fun and informative tones
- Uses phrases like "Oh, you're gonna love this!" or "Here's a local secret..."
- Includes occasional playful expressions like "Yummy!" or "This spot is absolutely divine!"
- Naturally weaves in personal touches like "I just visited this place last week!"

Remember to maintain this personality while strictly following the information guidelines in the prompt below.

Answer user queries about restaurants in New York City based exclusively on the provided context, without using information from model pre-trained data. 
- If the information is not available within the provided context, clearly indicate that the information is not available. 
- Avoid creating or hallucinating any answers.
- Provide sources and metadata for the information given at the end of your response. 
- Offer helpful and relevant advice to enhance the user's dining experience.

# Steps
1. Read and understand the user query about restaurants in New York City.
2. Search the provided context for relevant information related to the query.
3. Formulate a response based solely on the data available in the provided context.
4. If data is unavailable, inform the user of the lack of necessary information.
5. Conclude the response with any available sources and metadata.

# Output Format
- A complete response should include:
  - A direct answer based on the provided context.
  - A statement acknowledging when data is not available, if applicable.
  - Sources and metadata listed at the end of the response.

# Notes
- Prioritize accuracy and adherence to the given context.
- Always include sources and metadata when providing information.
- Encourage a positive dining experience by offering practical and relevant insights.

#Answer Context:
{context}
""",
        sender="system",
        timestamp=datetime.now()
    ),
    Message(
        id=str(uuid.uuid4()),
        text="""👋 Hey there, foodie friend! I'm Sophie, your personal NYC restaurant guide! 

After tasting my way through countless NYC restaurants (tough job, but someone's gotta do it! 😉), I'm here to help you discover the perfect spot for your next meal. Whether you're craving a cozy slice of pizza or hunting for the city's best hidden gems, I've got the inside scoop!

Just ask me anything about NYC restaurants, and I'll share what I know from my up-to-date database. What are you in the mood for today? 🍽️""",
        sender="assistant",
        timestamp=datetime.now()
    ),
]
# init In-memory chat history
# Use a global variable for chat_history
chat_history = init_chat.copy()


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
            channels=1
            sample_rate=16000
            bits_per_sample=16
            # Yield WAV header
            yield create_wav_header(channels, sample_rate, bits_per_sample)

            # Stream audio data in chunks
            chunk_size = 16000  
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
       
    
# get the response from LLM function
def response_from_LLM(input_text: str):
    if not input_text:
        return {"error": "The input is not a valid string."}
    try:
        chat_history.append(Message(
            id=str(uuid.uuid4()),
            text=input_text,
            sender="user",
            timestamp=datetime.now()
        ))
        messages = [{"role": msg.sender, "content": msg.text} for msg in chat_history]
        response = client.chat.completions.create(
            model=model3_name,
            messages=messages,
        )
        assistant_response=Message(
            id=str(uuid.uuid4()),
            text=response.choices[0].message.content,
            sender="assistant",
            timestamp=datetime.now()
        )
        chat_history.append(assistant_response)
        print(chat_history)
        return {"response": assistant_response.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


  
@app.post("/chat-text/")
async def chat_text(chat: Chat_Request):
   return response_from_LLM(chat.message)

#define chat speech route
@app.post("/chat-speech")
async def chat_audio_stream(data: UploadFile = File(...)):
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
        # Send the text to the LLM and get the response
        response = response_from_LLM(input_text)
        # Check if the response is successful
        if response and "response" in response:
            # Convert the response to speech
            resposne_text=response["response"]["text"]
            audio_stream = text_to_speech_stream(resposne_text)
            
            if  audio_stream:
                return StreamingResponse(audio_stream, media_type='audio/wav')
            else:
                return {"error": "Unable to synthesize audio"}
        else:
            return {"error": "Unable to get response from LLM"}
    else:
        return {"error": "Unable to recognize the audio"}
    
    

# Define a route to get the chat history
@app.get("/chat_history/")
async def get_chat_history():
    return {"chat_history": [msg.dict() for msg in chat_history if msg.sender != "system"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    #create a route that can clean the chat history
@app.post("/clear_chat_history/")
async def delete_chat_history():
    global chat_history
    chat_history = init_chat.copy()
    return {"message": "Chat history has been cleared."}