import React, { useEffect, useState } from 'react';
import { getTextResponse, getChatHistory } from './apiService';
import Speech from './Speech';
import axios from 'axios';
import ChatMessages from './ChatMessage';
import { Message } from './types';
import InputArea from './InputArea';
import ClearButton from './ClearButton';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const onSendMessage = async (input: string) => {
    const tempId = `temp-${Date.now()}`;
    const newUserMessage: Message = {
      id: tempId,
      text: input,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    // Immediately update the messages state with the user's input
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
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
    <div className='flex flex-col h-screen bg-gradient-to-br from-indigo-100 to-purple-100'>
      <header className='bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-4 shadow-lg'>
        <h1 className='text-2xl font-bold text-center'>
          New York Restaurants Assistant
        </h1>
      </header>
      <main className='flex-grow flex flex-col overflow-hidden'>
        <div className='flex-grow overflow-y-auto p-4'>
          <ChatMessages messages={messages} isLoading={isLoading} />
        </div>
      </main>
      <footer className='flex items-center bg-white border-t border-gray-200 p-4 shadow-inner'>
        <div className='max-w-4xl mx-auto flex items-center space-x-4'>
          <InputArea onSendMessage={onSendMessage} />

          <Speech setMessages={setMessages} getHistory={getHistory} />
        </div>
        <ClearButton onClear={clearChatHistory} />
      </footer>
    </div>
  );
};

export default ChatInterface;
