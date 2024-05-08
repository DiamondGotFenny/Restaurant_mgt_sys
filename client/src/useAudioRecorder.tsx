import { useState, useEffect } from 'react';

import { getWaveBlob } from 'webm-to-wav-converter';
interface UseAudioRecorderReturn {
  startRecording: () => void;
  stopRecording: () => void;
  audioBlob: Blob | null;
  isRecording: boolean;
  clearAudioBlob: () => void;
}

export const useAudioRecorder = (): UseAudioRecorderReturn => {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(
    null
  );
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    const constraints = { audio: { channelCount: 1 }, video: false };

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((stream) => {
        const newMediaRecorder = new MediaRecorder(stream);

        setMediaRecorder(newMediaRecorder);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);

  const startRecording = () => {
    if (mediaRecorder) {
      try {
        mediaRecorder.start();

        setIsRecording(true);
      } catch (error) {
        console.error('Error starting the audio recorder:', error);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          console.log(e.data, '  e.data');
          const wavBlob = getWaveBlob(e.data, false, {
            sampleRate: 16000,
          });
          wavBlob.then((blob) => {
            setAudioBlob(blob);
          });
        }
      };
      mediaRecorder.stop();
    }
    setIsRecording(false);
  };

  const clearAudioBlob = () => {
    setAudioBlob(null);
  };

  return {
    startRecording,
    stopRecording,
    audioBlob,
    isRecording,
    clearAudioBlob,
  };
};
