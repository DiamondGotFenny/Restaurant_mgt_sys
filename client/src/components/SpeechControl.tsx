import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Volume2Icon, VolumeOffIcon } from 'lucide-react';
import { useChatStore } from '../store/useChatStore';
import { useAudioRecorder } from '../useAudioRecorder';
import { cn } from '../lib/utils';
import { AudioLoadingPlayer } from '../lib/AudioLoadingPlayer';

const SpeechControl: React.FC = () => {
  const { setLoading, isLoading, getHistory } = useChatStore();
  const [isMuted, setIsMuted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const loadingPlayerRef = useRef<AudioLoadingPlayer | null>(null);

  const {
    startRecording,
    stopRecording,
    audioBlob,
    isRecording,
    clearAudioBlob,
  } = useAudioRecorder();

  useEffect(() => {
    loadingPlayerRef.current = new AudioLoadingPlayer();
    return () => {
      loadingPlayerRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (isLoading) {
      loadingPlayerRef.current?.start();
    } else {
      loadingPlayerRef.current?.stop();
    }
  }, [isLoading]);

  useEffect(() => {
    if (isMuted) {
      loadingPlayerRef.current?.setVolume(0);
    } else {
      loadingPlayerRef.current?.setVolume(1);
    }
  }, [isMuted]);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
      setLoading(true);
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
        await getHistory();
        clearAudioBlob();
      } catch (error) {
        console.error('Error sending audio data:', error);
      } finally {
        setLoading(false);
      }
    } else {
      console.log('no audio chunks or audio is not recorded correctly!');
      setLoading(false);
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
    <div className={cn('flex items-center gap-2')}>
      <button
        onClick={toggleMute}
        disabled={!isPlaying && !isLoading}
        className={cn(
          'w-10 h-10 rounded-full flex items-center justify-center',
          'transition-all duration-300 ease-in-out',
          isMuted ? 'bg-red-500' : 'bg-green-500 text-white',
          isPlaying || isLoading ? 'animate-pulse' : ''
        )}>
        {isMuted ? <VolumeOffIcon size={20} /> : <Volume2Icon size={20} />}
      </button>
      <button
        onClick={toggleRecording}
        disabled={isLoading || isPlaying}
        className={cn(
          'w-60 px-4 py-2 rounded-full border',
          'transition-all duration-300 ease-in-out',
          isLoading || isPlaying
            ? 'opacity-50 cursor-not-allowed'
            : 'hover:opacity-80',
          isRecording ? 'animate-pulse' : '',
          isRecording
            ? 'bg-red-50 border-red-500 text-red-700'
            : 'border-gray-300 hover:bg-gray-50'
        )}>
        {isRecording ? 'Recording...' : 'Click to record'}
      </button>
    </div>
  );
};

export default SpeechControl;
