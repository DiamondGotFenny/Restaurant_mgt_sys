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
  userInput: FormData,
  endpoint: string
): Promise<{ text: string; audio: string; input: string }> => {
  const config = {
    headers: {
      'content-type': 'multipart/form-data',
    },
  };
  try {
    //don't use axios , use fetch instead for multipart/form-data
    /* const response = await fetch(endpoint, {
      method: 'POST',
      body: userInput,
      headers: {
        'content-type': 'application/x-www-form-urlencoded',
      },
    }); */
    const response = await axios.post(endpoint, userInput, config);

    console.log(response);

    // Extracting text and audio data from the response
    const { response_text, response_audio, input_text } = response.data;

    // Assuming audio_data is a base64 encoded string
    const audioBlob = new Blob([response_audio], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);

    return { text: response_text, audio: audioUrl, input: input_text };
  } catch (error) {
    console.error(error, ' error from getAudioResponse');
    return { text: 'Error', audio: '', input: 'error' };
  }
};

const getChatHistory = async (endpoint: string): Promise<IChatHistory[]> => {
  const response = await axios.get(endpoint);
  return response.data.chat_history;
};

export { getTextResponse, getAudioResponse, getChatHistory };
