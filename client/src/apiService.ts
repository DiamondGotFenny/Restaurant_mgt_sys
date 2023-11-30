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

// Create a function that gets both text and audio response
const getAudioResponse = async (
  userInput: string,
  endpoint: string
): Promise<{ text: string; audio: string }> => {
  try {
    const response = await axios.post(endpoint, {
      message: userInput,
    });

    console.log(response);

    // Extracting text and audio data from the response
    const { text_data, audio_data } = response.data;

    // Assuming audio_data is a base64 encoded string
    const audioBlob = new Blob([audio_data], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);

    return { text: text_data, audio: audioUrl };
  } catch (error) {
    console.error(error);
    return { text: 'Error', audio: '' };
  }
};

const getChatHistory = async (endpoint: string): Promise<IChatHistory[]> => {
  const response = await axios.get(endpoint);
  return response.data.chat_history;
};

export { getTextResponse, getAudioResponse, getChatHistory };
