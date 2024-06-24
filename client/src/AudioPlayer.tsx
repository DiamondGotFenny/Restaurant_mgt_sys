import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Message } from './chatInterface';

// Create an audio player component that accepts a text string
interface AudioPlayerProps {
  text: string;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

const AudioPlayer = ({ text, setMessages, getHistory }: AudioPlayerProps) => {
  const [audioUrl, setAudioUrl] = useState<string | undefined>(undefined);

  //////need to add audio controls
  const fetchAudioData = async () => {
    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_BASE_URL}/chat-text-to-speech/`,
        { message: text },
        {
          responseType: 'arraybuffer',
          headers: { 'Content-Type': 'application/json' },
        }
      );

      const arrayBuffer = response.data;
      const audioContext = new window.AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start();
      await getHistory(setMessages);
    } catch (error) {
      console.error('Error fetching audio data:', error);
    }
  };
  ////////

  useEffect(() => {
    if (!text) return;
    async function fetchData() {
      try {
        const response = await axios.post<ArrayBuffer>(
          `${process.env.REACT_APP_API_BASE_URL}/chat-text-to-speech/`,
          { message: text },
          {
            responseType: 'arraybuffer',
            headers: { 'Content-Type': 'application/json' },
          }
        );
        //check the response.data if valid audio data
        const audioblob = new Blob([response.data], { type: 'audio/wav' });
        //check if the audioblob is a valid audio
        console.log(audioblob, ' audioblob');
        const myurl = URL.createObjectURL(audioblob);

        setAudioUrl(myurl);
        await getHistory(setMessages);
      } catch (error) {
        console.error('Error fetching audio:', error);
      }
    }
    //fetchData();
    fetchAudioData();
  }, [text]);
  return (
    <div>
      <audio controls src={audioUrl} autoPlay />
    </div>
  );
};

export default AudioPlayer;
