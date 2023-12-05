import { useEffect, useState } from 'react';
import { getTextResponse, getChatHistory } from './apiService';
import SpeechRecongnition from './SpeechRecongnition';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (input.trim() !== '') {
      setMessages((messages) => [
        ...messages,
        { role: 'user', content: input },
      ]);
      setInput('');
      const response = await getTextResponse(
        input,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text/`
      );
      setMessages((messages) => [
        ...messages,
        { role: 'assistant', content: response },
      ]);
    }
  };

  useEffect(() => {
    const getHistory = async () => {
      const response = await getChatHistory(
        `${process.env.REACT_APP_API_BASE_URL}/chat_history/`
      );
      setMessages([...response]);
    };
    getHistory();
  }, []);
  console.log(messages, ' message component');
  return (
    <div>
      <div>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{ textAlign: message.role === 'user' ? 'right' : 'left' }}>
            {message.content}
          </div>
        ))}
      </div>
      <input
        type='text'
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
      />
      <button onClick={sendMessage}>Send</button>
      <SpeechRecongnition setMessages={setMessages} />
    </div>
  );
};

export default ChatInterface;
