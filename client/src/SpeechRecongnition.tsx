import { useState, useEffect, useRef } from 'react';
import { Message } from './chatInterface';
import { getAudioResponse } from './apiService';
import { useReactMediaRecorder } from 'react-media-recorder-2';
//define the component prop
interface SpeechRecorderProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const SpeechRecongnition = ({ setMessages }: SpeechRecorderProps) => {
  const { status, startRecording, stopRecording, mediaBlobUrl } =
    useReactMediaRecorder({ audio: true, video: false });
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioUrl, setAudioUrl] = useState<string | undefined>(undefined);

  const toggleRecording = () => {
    if (status === 'recording') {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const sendDataToServer = async (mediaBlobUrl: string | undefined) => {
    if (mediaBlobUrl) {
      const myblob = await fetch(mediaBlobUrl);
      const audioBlob = await myblob.blob(); // Convert the data to a Blob
      // Create a FormData object
      const formData = new FormData();
      console.log(audioBlob, 'recordingBlob');
      // Append the audio data to the FormData object
      formData.append('data', audioBlob, 'recording.wav');
      console.log(formData.get('data'), 'formData');
      const response = await getAudioResponse(
        formData,
        `${process.env.REACT_APP_API_BASE_URL}/chat-voice/`
      );
      setMessages((messages) => [
        ...messages,
        { role: 'user', content: response.input },
      ]);
      setMessages((messages) => [
        ...messages,
        { role: 'assistant', content: response.text },
      ]);
      setAudioUrl(response.audio); // Set the audio URL
    } else {
      console.log('no audio chunks or audio is not recorded correctly!');
    }
  };

  //play the audio mediaBlobUrl is not null
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      const audio = audioRef.current;
      audio.src = audioUrl;
      audio.play();
    }
  }, [audioUrl]);
  //call the sendDataToServer function when mediaBlobUrl is not null
  useEffect(() => {
    if (mediaBlobUrl) {
      sendDataToServer(mediaBlobUrl);
    }
  }, [mediaBlobUrl]);
  return (
    <div>
      <p>{status}</p>
      <button onClick={toggleRecording}>
        {status === 'recording' ? 'Stop' : 'Start'}
      </button>
      <audio ref={audioRef} hidden />
    </div>
  );
};

export default SpeechRecongnition;
