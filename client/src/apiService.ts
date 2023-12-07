import axios from 'axios';

interface IChatHistory {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const getTextResponse = async (
  userInput: string,
  endpoint: string
): Promise<string> => {
  try {
    const response = await axios.post(endpoint, {
      message: userInput,
    });
    console.log(response);
    return response.data.response;
  } catch (error) {
    console.error(error);
    return 'Error';
  }
};

// Create a function that send user input text to the server and get the text and audio response
const sendTextToSpeechRequest = async (
  userInput: string,
  endpoint: string
): Promise<{ text: string; audio: string }> => {
  try {
    const response = await axios.post(endpoint, {
      message: userInput,
    });

    console.log(response, ' response from getAudioResponse');

    // Extracting text and audio data from the response
    const { text, audio } = response.data;

    // Assuming audio_data is a base64 encoded string
    const audioBlob = new Blob([audio], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);

    return { text: text, audio: audioUrl };
  } catch (error) {
    console.error(error, ' error from getAudioResponse');
    return {
      text: 'sorry I did not catch you, could please repeat it?',
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

const getChatHistory = async (endpoint: string): Promise<IChatHistory[]> => {
  const response = await axios.get(endpoint);
  return response.data.chat_history;
};

export {
  getTextResponse,
  sendTextToSpeechRequest,
  getChatHistory,
  sendSpeechToTextRequest,
};
