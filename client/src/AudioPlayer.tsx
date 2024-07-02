import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Message } from './types';
import { FaVolumeMute, FaVolumeUp } from 'react-icons/fa';
import styled, { keyframes } from 'styled-components';

// Create an audio player component that accepts a text string
interface AudioPlayerProps {
  text: string;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(0, 123, 255, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(0, 123, 255, 0);
  }
`;

interface MuteButtonProps {
  $isMuted: boolean;
  $isPlaying: boolean;
}

const MuteButton = styled.button.attrs<MuteButtonProps>(() => ({
  type: 'button',
}))<MuteButtonProps>`
  background-color: ${(props) => (props.$isMuted ? '#dc3545' : '#28a745')};
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  animation: ${(props) => (props.$isPlaying ? pulse : 'none')} 2s infinite;

  &:hover {
    opacity: 0.8;
  }

  &:focus {
    outline: none;
  }
`;

const AudioPlayer = ({ text, setMessages, getHistory }: AudioPlayerProps) => {
  const [isMuted, setIsMuted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  //////need to add styles to buttons

  ////////
  const handleMute = () => {
    if (gainNodeRef.current && audioContextRef.current) {
      if (isMuted) {
        console.log('unmute');
        gainNodeRef.current.gain.setValueAtTime(
          1,
          audioContextRef.current.currentTime
        );
      } else {
        console.log('mute');
        gainNodeRef.current.gain.setValueAtTime(
          0,
          audioContextRef.current.currentTime
        );
      }
      setIsMuted(!isMuted);
    }
  };
  useEffect(() => {
    if (!text) return;
    const fetchAudioData = async () => {
      if (!text) return;

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
        audioContextRef.current = new window.AudioContext();
        const audioBuffer = await audioContextRef.current.decodeAudioData(
          arrayBuffer
        );

        sourceNodeRef.current = audioContextRef.current.createBufferSource();
        sourceNodeRef.current.buffer = audioBuffer;

        gainNodeRef.current = audioContextRef.current.createGain();
        sourceNodeRef.current.connect(gainNodeRef.current);
        gainNodeRef.current.connect(audioContextRef.current.destination);

        sourceNodeRef.current.start();
        setIsPlaying(true);

        sourceNodeRef.current.onended = () => setIsPlaying(false);

        await getHistory(setMessages);
      } catch (error) {
        console.error('Error fetching audio data:', error);
      }
    };
    fetchAudioData();
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [text]);
  return (
    <div className='audio-player'>
      <div className='controls'>
        <MuteButton
          onClick={handleMute}
          $isMuted={isMuted}
          $isPlaying={isPlaying}>
          {isMuted ? <FaVolumeMute size={24} /> : <FaVolumeUp size={24} />}
        </MuteButton>
      </div>
    </div>
  );
};

export default AudioPlayer;
