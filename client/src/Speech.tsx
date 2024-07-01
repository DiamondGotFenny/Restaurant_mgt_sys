import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Message } from './chatInterface';
import { FaMicrophone, FaStop, FaVolumeMute, FaVolumeUp } from 'react-icons/fa';
import styled, { keyframes, css } from 'styled-components';
import { useAudioRecorder } from './useAudioRecorder';

interface CombinedSpeechProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

const wave = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
`;

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(0, 123, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }
`;

const Button = styled.button<{ $disabled: boolean }>`
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: ${(props) => (props.$disabled ? 'not-allowed' : 'pointer')};
  transition: all 0.3s ease;
  margin: 0 10px;
  opacity: ${(props) => (props.$disabled ? 0.5 : 1)};

  &:hover {
    opacity: ${(props) => (props.$disabled ? 0.5 : 0.8)};
  }

  &:focus {
    outline: none;
  }
`;

const RecordButton = styled(Button)<{ $isRecording: boolean }>`
  background-color: ${(props) => (props.$isRecording ? '#dc3545' : '#28a745')};
  animation: ${(props) => (props.$isRecording ? pulse : 'none')} 2s infinite;
`;

const MuteButton = styled(Button)<{ $isMuted: boolean; $isPlaying: boolean }>`
  background-color: ${(props) => (props.$isMuted ? '#dc3545' : '#28a745')};
  animation: ${(props) =>
    props.$isPlaying && !props.$isMuted
      ? css`
          ${wave} 1s infinite
        `
      : 'none'};
`;

const ButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 20px;
`;

const Speech: React.FC<CombinedSpeechProps> = ({ setMessages, getHistory }) => {
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
      // Create a FormData object
      const formData = new FormData();
      // Append the audio data to the FormData object
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
    }
  };

  //call the sendDataToServer function when mediaBlobUrl is not null
  useEffect(() => {
    if (audioBlob) {
      //create the audioBlob to audio/wav type blob because the audioBlob is audio/wave type
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
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <ButtonContainer>
      <RecordButton
        onClick={toggleRecording}
        $isRecording={isRecording}
        $disabled={isPlaying || isWaiting}
        disabled={isPlaying || isWaiting}>
        {isRecording ? <FaStop size={24} /> : <FaMicrophone size={24} />}
      </RecordButton>
      <MuteButton
        onClick={toggleMute}
        $isMuted={isMuted}
        $isPlaying={isPlaying}
        $disabled={!isPlaying}
        disabled={!isPlaying}>
        {isMuted ? <FaVolumeMute size={24} /> : <FaVolumeUp size={24} />}
      </MuteButton>
    </ButtonContainer>
  );
};

export default Speech;
