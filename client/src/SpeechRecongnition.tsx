import { useState, useEffect, useRef } from 'react';
import { Message } from './chatInterface';
import { getAudioResponse } from './apiService';
import { useAudioRecorder } from './useAudioRecorder';
//define the component prop
interface SpeechRecorderProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const SpeechRecongnition = ({ setMessages }: SpeechRecorderProps) => {
  /***
   * seems the speech converation is not as smooth as the we connect speech to text Azure service
   * directly in frontend, not sure it has something to do with we record the vocie at client side first,
   * then send it to our server, then use the Azure speech to text service. need more experiments
   */
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
   * todo: get the user input text from another sperate route, return the text after the speech recognition,
   * no need to wait for the audio response, as it make take a while to get the response from the LLM
   */
  const sendDataToServer = async (audioBlob: Blob | null) => {
    if (audioBlob) {
      // Create a FormData object
      const formData = new FormData();
      // Append the audio data to the FormData object
      formData.append('data', audioBlob, 'recording.wav');
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
