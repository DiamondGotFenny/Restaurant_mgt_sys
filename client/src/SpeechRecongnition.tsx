import { useState, useEffect, useRef } from 'react';
import { Message } from './chatInterface';
import { sendTextToSpeechRequest, sendSpeechToTextRequest } from './apiService';
import { useAudioRecorder } from './useAudioRecorder';
//define the component prop
interface SpeechRecorderProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const SpeechRecongnition = ({ setMessages }: SpeechRecorderProps) => {
  const { startRecording, stopRecording, audioBlob, isRecording } =
    useAudioRecorder();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioUrl, setAudioUrl] = useState<string | undefined>(undefined);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };
  /***
   *use the stream to send the speech back
   */
  const sendDataToServer = async (audioBlob: Blob | null) => {
    if (audioBlob) {
      // Create a FormData object
      const formData = new FormData();
      // Append the audio data to the FormData object
      formData.append('data', audioBlob, 'recording.wav');
      const response = await sendSpeechToTextRequest(
        formData,
        `${process.env.REACT_APP_API_BASE_URL}/chat-speech-to-text/`
      );
      setMessages((messages) => [
        ...messages,
        { role: 'user', content: response },
      ]);
      const responseAudio = await sendTextToSpeechRequest(
        response,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text-to-speech/`
      );
      setMessages((messages) => [
        ...messages,
        { role: 'assistant', content: responseAudio.text },
      ]);
      setAudioUrl(responseAudio.audio); // Set the audio URL
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
    if (audioBlob) {
      sendDataToServer(audioBlob);
    }
  }, [audioBlob]);
  return (
    <div>
      <button onClick={toggleRecording}>
        {isRecording ? 'Stop' : 'Start'}
      </button>
      <audio ref={audioRef} hidden />
    </div>
  );
};

export default SpeechRecongnition;
