//apiService.ts
import axios from 'axios';
import { Message, Promotion, RawRoutingResult } from './types';

interface ChatResponse {
  response: Message;
  raw_routing_result: RawRoutingResult;
}

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
): Promise<ChatResponse> => {
  try {
    const response = await axios.post(endpoint, {
      message: userInput,
    });
    console.log(response);
    return response.data;
  } catch (error) {
    console.error(error);
    return {
      response: {
        id: generateErrorId(),
        text: 'Sorry, there is an error, please try again.',
        sender: 'assistant',
        timestamp: generateErrorTimestamp(),
      },
      raw_routing_result: {
        is_relevant: false,
        vector_search_result: null,
        sql_result: null,
      },
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
  getChatHistory,
  sendSpeechToTextRequest,
  getPromotions,
};
