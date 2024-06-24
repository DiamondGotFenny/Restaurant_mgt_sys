import { useState, useEffect } from 'react';
import { Message } from './chatInterface';
import { sendSpeechToTextRequest } from './apiService';
import { useAudioRecorder } from './useAudioRecorder';
import AudioPlayer from './AudioPlayer';

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

  const [text, setText] = useState<string>(''); // State to store the text response

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
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
      const response = await sendSpeechToTextRequest(
        formData,
        `${process.env.REACT_APP_API_BASE_URL}/chat-speech-to-text/`
      );
      await getHistory(setMessages);
      setText(response);
      clearAudioBlob();
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

  return (
    <div>
      <button onClick={toggleRecording}>
        {isRecording ? 'Stop' : 'Start'}
      </button>
      <AudioPlayer
        text={text}
        setMessages={setMessages}
        getHistory={getHistory}
      />
    </div>
  );
};

export default SpeechRecongnition;
