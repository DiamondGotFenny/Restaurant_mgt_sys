import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Message } from './types';
import { FaMicrophone, FaStop, FaVolumeMute, FaVolumeUp } from 'react-icons/fa';
import { useAudioRecorder } from './useAudioRecorder';

interface SpeechProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

const Speech: React.FC<SpeechProps> = ({ setMessages, getHistory }) => {
  const [isMuted, setIsMuted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isWaiting, setIsWaiting] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);

  const {
    startRecording,
    stopRecording,
    audioBlob,
    isRecording,
    clearAudioBlob,
  } = useAudioRecorder();

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
      setIsWaiting(true);
    } else {
      startRecording();
    }
  };

  const sendDataToServer = async (audioBlob: Blob | null) => {
    if (audioBlob) {
      const formData = new FormData();
      formData.append('data', audioBlob, 'recording.wav');
      try {
        const response = await axios.post(
          `${process.env.REACT_APP_API_BASE_URL}/chat-speech`,
          formData,
          {
            responseType: 'arraybuffer',
            headers: {
              'content-type': 'multipart/form-data',
            },
          }
        );

        playAudioResponse(response.data);
        await getHistory(setMessages);
        clearAudioBlob();
      } catch (error) {
        console.error('Error sending audio data:', error);
      } finally {
        setIsWaiting(false);
      }
    } else {
      console.log('no audio chunks or audio is not recorded correctly!');
      setIsWaiting(false);
    }
  };

  useEffect(() => {
    if (audioBlob) {
      const newBlob = new Blob([audioBlob], { type: 'audio/wav' });
      sendDataToServer(newBlob);
    }
  }, [audioBlob]);

  const playAudioResponse = async (arrayBuffer: ArrayBuffer) => {
    audioContextRef.current = new window.AudioContext();
    const audioBuffer = await audioContextRef.current.decodeAudioData(
      arrayBuffer
    );

    sourceNodeRef.current = audioContextRef.current.createBufferSource();
    sourceNodeRef.current.buffer = audioBuffer;

    gainNodeRef.current = audioContextRef.current.createGain();
    sourceNodeRef.current.connect(gainNodeRef.current);
    gainNodeRef.current.connect(audioContextRef.current.destination);

    if (isMuted) {
      gainNodeRef.current.gain.setValueAtTime(
        0,
        audioContextRef.current.currentTime
      );
    }

    sourceNodeRef.current.start();
    setIsPlaying(true);

    sourceNodeRef.current.onended = () => {
      setIsPlaying(false);
    };
  };

  const toggleMute = () => {
    if (gainNodeRef.current && audioContextRef.current) {
      if (isMuted) {
        gainNodeRef.current.gain.setValueAtTime(
          1,
          audioContextRef.current.currentTime
        );
      } else {
        gainNodeRef.current.gain.setValueAtTime(
          0,
          audioContextRef.current.currentTime
        );
      }
      setIsMuted(!isMuted);
    }
  };

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <div className='flex justify-center mt-5'>
      <button
        onClick={toggleRecording}
        disabled={isPlaying || isWaiting}
        className={`
          w-12 h-12 rounded-full flex items-center justify-center
          transition-all duration-300 ease-in-out mr-2
          ${
            isRecording
              ? 'bg-red-500 text-white animate-pulse'
              : 'bg-green-500 text-white'
          }
          ${
            isPlaying || isWaiting
              ? 'opacity-50 cursor-not-allowed'
              : 'hover:opacity-80'
          }
        `}>
        {isRecording ? <FaStop size={24} /> : <FaMicrophone size={24} />}
      </button>
      <button
        onClick={toggleMute}
        disabled={!isPlaying}
        className={`
          w-12 h-12 rounded-full flex items-center justify-center
          transition-all duration-300 ease-in-out
          ${isMuted ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}
          ${!isPlaying ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-80'}
          ${isPlaying && !isMuted ? 'animate-pulse' : ''}
        `}>
        {isMuted ? <FaVolumeMute size={24} /> : <FaVolumeUp size={24} />}
      </button>
    </div>
  );
};

export default Speech;
