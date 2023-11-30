import { useState, useEffect, useRef } from 'react';
import * as sppechSdk from 'microsoft-cognitiveservices-speech-sdk';
import getMicrophone from './MicrophoneAccess';
import { Message } from './chatInterface';
import { getAudioResponse } from './apiService';
//define the component prop
interface SpeechRecorderProps {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

const SpeechRecongnition = ({ setMessages }: SpeechRecorderProps) => {
  //creadentials for azure speech
  const speech_key = process.env.REACT_APP_SPEECH_API_KEY || '';
  const service_region = process.env.REACT_APP_SPEECH_API_REGION || '';
  //const speech_endpoint = process.env.REACT_APP_SPEECH_API_ENDPOINT || '';

  //record speech from microphone then send to azure speech to text
  const speechConfig = sppechSdk.SpeechConfig.fromSubscription(
    speech_key,
    service_region
  );
  speechConfig.speechRecognitionLanguage = 'en-US';
  const audioConfig = sppechSdk.AudioConfig.fromDefaultMicrophoneInput();
  const recognizer = new sppechSdk.SpeechRecognizer(speechConfig, audioConfig);

  const [status, setStatus] = useState('stopped');
  const [isRecording, setIsRecording] = useState(false);

  const [audioUrl, setAudioUrl] = useState('');
  const audioRef = useRef<HTMLAudioElement>(null);

  const sendMessage = async (input_text: string) => {
    if (input_text.trim() !== '') {
      setMessages((messages) => [
        ...messages,
        { role: 'user', content: input_text },
      ]);
      const response = await getAudioResponse(
        input_text,
        `${process.env.REACT_APP_API_BASE_URL}/chat-voice/`
      );
      setMessages((messages) => [
        ...messages,
        { role: 'assistant', content: response.text },
      ]);
      setAudioUrl(response.audio); // Set the audio URL
    }
  };

  //function to recongize the speech
  const startRecongnizing = () => {
    setStatus('recording');
    recognizer.recognizeOnceAsync(
      (result) => {
        if (result.reason === sppechSdk.ResultReason.RecognizedSpeech) {
          sendMessage(result.text);
          setStatus('stopped');
        } else {
          console.log(
            'No speech could be recognized or no speech was detected.'
          );
          setStatus('stopped');
        }
        setIsRecording(false);
        recognizer.close();
      },
      (err) => {
        console.error('ERROR: ' + err);
        setIsRecording(false);
        setStatus('stopped');
        recognizer.close();
      }
    );
  };

  const toggleRecording = () => {
    if (!isRecording) {
      setIsRecording(true);
      getMicrophone().then(startRecongnizing).catch(console.error);
    } else {
      setStatus('stopped');
      recognizer.stopContinuousRecognitionAsync(
        () => {
          recognizer.close();
        },
        (err) => {
          console.error('ERROR: ' + err);
          recognizer.close();
        }
      );
    }
  };

  // Play audio when the audioUrl state changes
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.play().catch(console.error);
    }
  }, [audioUrl]);

  return (
    <div>
      <p>{status}</p>
      <button onClick={toggleRecording} disabled={isRecording}>
        {isRecording ? 'Stop' : 'Start'}
      </button>
      <audio ref={audioRef} hidden />
    </div>
  );
};

export default SpeechRecongnition;
