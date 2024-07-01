import { useEffect, useState } from 'react';
import { getTextResponse, getChatHistory } from './apiService';
import SpeechRecongnition from './SpeechRecongnition';
import Speech from './Speech';
import axios from 'axios';

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

  const getHistory = async (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => {
    const response = await getChatHistory(
      `${process.env.REACT_APP_API_BASE_URL}/chat_history/`
    );
    setMessages([...response]);
  };

  useEffect(() => {
    getHistory(setMessages);
  }, []);
  console.log(messages, ' message component');

  //make axios request to clear chat history
  const clearChatHistory = async () => {
    await axios.post(
      `${process.env.REACT_APP_API_BASE_URL}/clear_chat_history/`
    );
    getHistory(setMessages);
    console.log('chat history cleared');
  };

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
      <Speech setMessages={setMessages} getHistory={getHistory} />
      <button onClick={() => clearChatHistory()}>Reset Chat</button>
    </div>
  );
};

export default ChatInterface;
