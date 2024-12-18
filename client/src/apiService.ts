import axios from 'axios';
import { Message, Promotion } from './types';

function generateErrorId() {
  // Generate a random number between 1000 and 9999
  const id = Math.floor(Math.random() * 9000) + 1000;
  return id.toString();
}

function generateErrorTimestamp() {
  // Get the current date and time
  const now = new Date();
  // Format the timestamp as "YYYY-MM-DD HH:MM:SS"
  const timestamp = now.toISOString().replace(/T/, ' ').replace(/\..+/, '');
  return timestamp;
}

const getTextResponse = async (
  userInput: string,
  endpoint: string
): Promise<Message> => {
  try {
    const response = await axios.post(endpoint, {
      message: userInput,
    });
    console.log(response);
    return response.data.response;
  } catch (error) {
    console.error(error);
    return {
      id: generateErrorId(),
      text: 'Sorry, there is an error, please try again.',
      sender: 'assistant',
      timestamp: generateErrorTimestamp(),
    };
  }
};

// Create a function that send user input text to the server and get the text and audio response
const sendTextToSpeechRequest = async (
  userInput: string,
  endpoint: string
): Promise<{ audio: string }> => {
  try {
    const response = await axios.post(
      endpoint,
      {
        message: userInput,
      },
      {
        responseType: 'arraybuffer', // Set the response type to handle binary data
      }
    );

    console.log(response, ' response from getAudioResponse');

    // Extracting text and audio data from the response
    const audioData = response.data;
    //check the type of audio
    console.log(typeof audioData, ' type of audio');

    return { audio: audioData };
  } catch (error) {
    console.error(error, ' error from getAudioResponse');
    return {
      audio: '',
    };
  }
};

//create a function that send speech data to the server and get the text response
const sendSpeechToTextRequest = async (
  speechData: FormData,
  endpoint: string
): Promise<string> => {
  const config = {
    headers: {
      'content-type': 'multipart/form-data',
    },
  };
  try {
    const response = await axios.post(endpoint, speechData, config);
    console.log('input text: ', response);
    return response.data.input_text;
  } catch (error) {
    console.error(error, ' error from sendSpeechToTextRequest');
    return 'sorry I did not catch you, could please repeat it?';
  }
};

const getChatHistory = async (endpoint: string): Promise<Message[]> => {
  const response = await axios.get(endpoint);
  return response.data.chat_history;
};

/**
 * Fetches promotional content to display while processing queries
 * @returns Promise containing an array of promotions
 */
const getPromotions = async (endpoint: string): Promise<Promotion[]> => {
  try {
    const response = await axios.get<{ promotions: Promotion[] }>(endpoint);
    return response.data.promotions;
  } catch (error) {
    console.error('Error fetching promotions:', error);
    return [];
  }
};

export {
  getTextResponse,
  sendTextToSpeechRequest,
  getChatHistory,
  sendSpeechToTextRequest,
  getPromotions,
};
