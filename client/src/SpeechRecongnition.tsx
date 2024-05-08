import { useState, useEffect, useRef } from 'react';
import { Message } from './chatInterface';
import { sendTextToSpeechRequest, sendSpeechToTextRequest } from './apiService';
import { useAudioRecorder } from './useAudioRecorder';

//define the component prop
interface SpeechRecorderProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

const SpeechRecongnition = ({
  setMessages,
  getHistory,
}: SpeechRecorderProps) => {
  const {
    startRecording,
    stopRecording,
    audioBlob,
    isRecording,
    clearAudioBlob,
  } = useAudioRecorder();

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
   * todo: solve the error:"Uncaught (in promise) DOMException: Failed to load because no supported source was found."
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
      const responseAudio = await sendTextToSpeechRequest(
        response,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text-to-speech/`
      );
      setAudioUrl(responseAudio.audio); // Set the audio URL
      await getHistory(setMessages);
    } else {
      console.log('no audio chunks or audio is not recorded correctly!');
    }
  };

  //play the audio mediaBlobUrl is not null
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      const audio = audioRef.current;
      audio.src = audioUrl;

      const audioPromise = audio.play();
      if (audioPromise !== undefined) {
        audioPromise
          .then(() => {
            console.log('play success');
          })
          .catch((error) => {
            console.log('play error', error);
            console.log(`Failed audio URL: ${audioUrl}`); // Log the problematic URL
          });
      }

      audio.onended = () => {
        setAudioUrl(undefined); // Clear the audioUrl state
        audio.src = ''; // Clear the audio element's src
        clearAudioBlob(); // Clear the audio blob
      };
    }
  }, [audioUrl]);
  //call the sendDataToServer function when mediaBlobUrl is not null
  useEffect(() => {
    if (audioBlob) {
      //create the audioBlob to audio/wav type blob because the audioBlob is audio/wave type
      const newBlob = new Blob([audioBlob], { type: 'audio/wav' });
      //check if the audioBlob is wav format
      if (newBlob.type !== 'audio/wav') {
        console.log(newBlob.type, 'The audio is not in wav format');
        return;
      }
      sendDataToServer(newBlob);
    }
  }, [audioBlob]);

  return (
    <div>
      <button onClick={toggleRecording}>
        {isRecording ? 'Stop' : 'Start'}
      </button>
      <audio ref={audioRef} controls autoPlay />
    </div>
  );
};

export default SpeechRecongnition;
