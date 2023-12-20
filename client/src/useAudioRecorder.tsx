import { useState, useEffect, useCallback } from 'react';

import { MediaRecorder, register } from 'extendable-media-recorder';
import { connect } from 'extendable-media-recorder-wav-encoder';

interface UseAudioRecorderReturn {
  startRecording: () => void;
  stopRecording: () => void;
  audioBlob: Blob | null;
  isRecording: boolean;
}
//put the register here, otherwise it will throw double regiger recorder error
register(await connect());

export const useAudioRecorder = (): UseAudioRecorderReturn => {
  const [mediaRecorder, setMediaRecorder] = useState<InstanceType<
    typeof MediaRecorder
  > | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [init, setInit] = useState(false);

  const setupRecorder = useCallback(async () => {
    try {
      const constraints: MediaStreamConstraints = {
        audio: true,
        video: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const audioContext = new AudioContext({
        sampleRate: 16000,
      });
      const mediaStreamAudioSourceNode = new MediaStreamAudioSourceNode(
        audioContext,
        { mediaStream: stream }
      );
      const mediaStreamAudioDestinationNode =
        new MediaStreamAudioDestinationNode(audioContext, {
          channelCount: 1,
        });
      mediaStreamAudioSourceNode.connect(mediaStreamAudioDestinationNode);

      const recorder = new MediaRecorder(
        mediaStreamAudioDestinationNode.stream,
        {
          mimeType: 'audio/wav',
        }
      );

      recorder.ondataavailable = (event: BlobEvent) => {
        setAudioBlob(event.data);
      };

      setMediaRecorder(recorder);
    } catch (error) {
      console.error('Error setting up the audio recorder:', error);
    }
  }, []);

  useEffect(() => {
    // avoid re-registering the encoder
    if (init) {
      return;
    }

    const setup = async () => {
      try {
        await setupRecorder();
      } catch (e) {
        console.error('Error setting up the audio recorder:', e);
      }
    };

    setup();
    setInit(true);
  }, []);

  const startRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
      mediaRecorder.start();
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  return { startRecording, stopRecording, audioBlob, isRecording };
};
