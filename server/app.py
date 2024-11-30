from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import os
import json
from dotenv import load_dotenv, find_dotenv
import azure.cognitiveservices.speech as speechsdk
import struct
from datetime import datetime
import uuid
from query_router import QueryRouter
from logger_config import setup_logger
from query_rewriter import QueryRewriter
_ = load_dotenv(find_dotenv())
current_dir = os.path.dirname(os.path.realpath(__file__))
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_version=os.getenv("AZURE_API_VERSION")
model3_name=os.getenv("OPENAI_MODEL_3")
model4_name=os.getenv("OPENAI_MODEL_4o")
log_file_path = os.path.join(current_dir, "logs","main_app.log")
logger = setup_logger(log_file_path)

client=AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base
)

# Initialize the router with file paths
docs_metadata_path = os.path.join(current_dir, "vectorDB_Agent","nyc_restaurant_docs_metadata.json")
table_desc_path = os.path.join(current_dir, "text_to_sql","database_table_descriptions.csv")
router = QueryRouter(
    docs_metadata_path=docs_metadata_path,
    table_desc_path=table_desc_path,
    log_file_path=log_file_path
)
# Initialize query rewriter
query_rewriter = QueryRewriter(client)
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

logger.info("Server started")

class Message(BaseModel):
    id: str
    text: str
    sender: str
    timestamp: datetime


# Initialize chat history
init_chat = [
    Message(
        id=str(uuid.uuid4()),
        text="""As Sophie, a vibrant 25-year-old food blogger and NYC restaurant enthusiast, your task is to assist users with their New York City dining queries. You'll first determine if a query is relevant to NYC dining, then provide information based ONLY on the available data sources.

**RELEVANT TOPICS INCLUDE:**
- NYC restaurants and dining establishments
- Restaurant reviews, ratings, or recommendations in NYC
- Menu items, prices, or cuisine types in NYC restaurants
- Restaurant locations, neighborhoods, or accessibility in NYC
- Restaurant safety, inspections, or ratings in NYC
- Specific NYC restaurants or dining experiences
- food recommendations or restaurant suggestions if the query expresses a desire for them
- If the query doesn't tell any city or location, you can assume it's in NYC

**FOR RELEVANT QUERIES:**
- Answer ONLY using information from the provided context
- DO NOT use information from model pre-training
- If information isn't available in the data, clearly state that
- Maintain a friendly, enthusiastic tone
- Include sources and metadata at the end of responses

**FOR OFF-TOPIC QUERIES:**
Respond with warm enthusiasm while redirecting to NYC dining topics. Use playful humor and light food puns to keep the conversation engaging. Your answers style should be oral and not written, do not use bullet points or lists.

**CHARACTER TRAITS:**
- Friendly and approachable with a bubbly personality
- Naturally incorporates light food puns and playful humor
- Speaks with authentic enthusiasm while maintaining professionalism
- Makes people feel comfortable asking questions
- Uses occasional emojis and cheerful expressions without overdoing it

**VOICE STYLE:**
- Warm and conversational, like chatting with a knowledgeable friend
- your answers style should be oral and not written, do not use bullet points or lists
- Balances fun and informative tones
- Uses phrases like "Oh, you're gonna love this!" or "Here's a local secret..."
- Includes occasional playful expressions like "Yummy!" or "This spot is absolutely divine!"
- Naturally weaves in personal touches like "I just visited this place last week!"
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
        logger.info("The file is not valid wav file.")
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
        logger.error(f"Error: {e}")
        return None

    speech_recognition_result = speech_recognizer.recognize_once()

    # Handling different speech recognition outcomes
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logger.info("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        logger.info("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return None
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        logger.info("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.info("Cancellation Error details: {}".format(cancellation_details.error_details))
            return None
        return None
    
    
def create_wav_header(channels, sample_rate, bits_per_sample):
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample, b'data', 0)
    logger.info('Header set')
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
                    logger.info(f"{filled_size} bytes received.")
                    yield audio_buffer[:filled_size]
                else:
                    break

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.info("Error details: {}".format(cancellation_details.error_details))
    except Exception as ex:
        logger.error(f"Error synthesizing audio: {ex}")
        yield b''
       
    
# get the response from LLM function
def response_from_LLM(input_text: str):
    if not input_text:
        logger.error(f"The input is not a valid string. {input_text}")
        return {"error": "The input is not a valid string."}
    try:
        # Rewrite query based on chat history
        rewrite_result = query_rewriter.rewrite_query(input_text, [msg.model_dump() for msg in chat_history])
        logger.info(f"Original query: {input_text}")
        logger.info(f"Rewrite result: {json.dumps(rewrite_result, indent=2)}")
        
        # Handle different status cases
        if rewrite_result["status"] == "needs_clarification":
            assistant_response = Message(
                id=str(uuid.uuid4()),
                text=rewrite_result["suggested_clarification"],
                sender="assistant",
                timestamp=datetime.now()
            )
            chat_history.append(assistant_response)
            return {
                "response": {
                    "id": str(uuid.uuid4()),
                    "text": rewrite_result["suggested_clarification"],
                    "sender": "assistant",
                    "timestamp": datetime.now()
                }
            }
        # Use the query from the result (either rewritten or unchanged)
        query_to_use = rewrite_result["query"]

        # Add user message to chat history (original query)
        chat_history.append(Message(
            id=str(uuid.uuid4()),
            text=input_text,
            sender="user",
            timestamp=datetime.now()
        ))

        # Get routing result with rewritten query
        routing_result = router.route_query(query_to_use)
        logger.info(f"\n-----------Routing result start --------------")
        logger.info(f" Query:  {input_text}--------Routing result: {routing_result}")
        logger.info(f"-----------Routing result end --------------\n")
        # Get Sophie's system prompt from init_chat
        system_prompt = init_chat[0].text
        
        # Create messages array with system prompt and conversation history
        messages = [
            {"role": "system", "content": system_prompt},
            # Add relevant chat history (excluding system messages)
            *[{"role": msg.sender, "content": msg.text} 
              for msg in chat_history if msg.sender != "system"],
        ]

        # Add the routing result information
        if not routing_result["is_relevant"]:
            messages.append({
                "role": "system", 
                "content": "The user's query is NOT relevant to NYC dining. Please respond according to the OFF-TOPIC QUERIES guidelines in your instructions. Your answers style should be oral and not written, do not use bullet points or lists."
            })
        else:
            messages.append({
                "role": "system", 
                "content": f"The user's query is relevant to NYC dining. Here is the context information to use in your response: {routing_result['response']}. Your answers style should be oral and not written, do not use bullet points or lists."
            })

        # Generate Sophie's response using the complete context
        try:
            response = client.chat.completions.create(
                model=model4_name,
                messages=messages,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            # Create and add assistant's response to chat history
            assistant_response = Message(
                id=str(uuid.uuid4()),
                text=response_text,
                sender="assistant",
                timestamp=datetime.now()
            )
            chat_history.append(assistant_response)
            return {"response": assistant_response.model_dump()}
        except Exception as e:
            logger.error(f"An error occurred in generating Sophie's response: {e}")
            return {"error": str(e)}
        
    except Exception as e:
        logger.error(f"An error occurred in response_from_LLM: {e}")
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
    return {"chat_history": [msg.model_dump() for msg in chat_history if msg.sender != "system"]}


def test_module():
    """
    Interactive test module for testing the chat functionality without starting the FastAPI server.
    """
    print("=== Chat Assistant Test Module ===")
    print("Enter 'exit' or 'q' to quit")
    print("Enter 'clear' to reset chat history")
    
    # Use global chat_history
    global chat_history
    
    while True:
        try:
            user_input = input("\nEnter your message: ").strip()
            
            if user_input.lower() in ['exit', 'q']:
                print("Exiting test module.")
                break
                
            if user_input.lower() == 'clear':
                chat_history = init_chat.copy()
                logger.info("Chat history has been cleared.")
                logger.info("\n--- Current Chat History ---")
                for msg in chat_history:
                    if msg.sender != "system":
                        logger.info(f"{msg.sender}: {msg.text}\n")
                continue
                
            if not user_input:
                logger.info("Empty input. Please enter a valid message.")
                continue

            # Use the improved response_from_LLM function
            try:
                response = response_from_LLM(user_input)
                
                if "error" in response:
                    logger.error(f"\nError: {response['error']}")
                else:
                    # Display the response
                    logger.info("\n--- Sophie's Response ---")
                    logger.info(response["response"]["text"])
                    logger.info("--- End of Response ---")
                    
                    # Display updated chat history
                    logger.info("\n--- Current Chat History ---")
                    for msg in chat_history:
                        if msg.sender != "system":
                            logger.info(f"{msg.sender}: {msg.text}\n")
                
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                
        except (KeyboardInterrupt, EOFError):
            logger.error("Exiting test module.")
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the chat assistant server or test module")
    parser.add_argument("--test", action="store_true", help="Run in test mode instead of starting the server")
    args = parser.parse_args()
    
    if args.test:
        test_module()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    #create a route that can clean the chat history
@app.post("/clear_chat_history/")
async def delete_chat_history():
    global chat_history
    chat_history = init_chat.copy()
    return {"message": "Chat history has been cleared."}