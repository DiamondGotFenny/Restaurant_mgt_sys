import { useEffect, useState } from 'react';
import { getTextResponse, getChatHistory } from './apiService';
import Speech from './Speech';
import axios from 'axios';
import ChatMessages from './ChatMessage';
import { Message } from './types';
import InputArea from './InputArea';
import ClearButton from './ClearButton';

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const onSendMessage = async (input: string) => {
    setIsLoading(true);
    try {
      await getTextResponse(
        input,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text/`
      );
      await getHistory(setMessages);
      setIsLoading(false);
    } catch (error) {
      console.log(error);
    } finally {
      setIsLoading(false);
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
    await getHistory(setMessages);
    console.log('chat history cleared');
  };

  return (
    <div className='flex flex-col h-screen bg-gray-100'>
      <header className='bg-blue-500 text-white p-4'>
        <h1 className='text-2xl font-bold'>AI Chat Assistant</h1>
      </header>
      <main className='flex-grow overflow-hidden'>
        <ChatMessages messages={messages} isLoading={isLoading} />
      </main>
      <footer className='bg-white border-t'>
        <div className='flex justify-between items-center p-4'>
          <div className='flex space-x-2'>
            <Speech setMessages={setMessages} getHistory={getHistory} />
          </div>
          <InputArea onSendMessage={onSendMessage} />
          <ClearButton onClear={clearChatHistory} />
        </div>
      </footer>
    </div>
  );
};

export default ChatInterface;
