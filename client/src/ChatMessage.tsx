// ChatMessages.tsx
import React from 'react';
import { Message } from './types';

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
}

const ChatMessages: React.FC<ChatMessagesProps> = ({ messages, isLoading }) => {
  console.log(isLoading);
  return (
    <div className='flex flex-col space-y-4 p-4 h-[calc(100vh-180px)] overflow-y-auto'>
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${
            message.sender === 'user' ? 'justify-end' : 'justify-start'
          }`}>
          <div
            className={`max-w-xs lg:max-w-md xl:max-w-lg px-4 py-2 rounded-lg ${
              message.sender === 'user'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-800'
            }`}>
            {message.text}
          </div>
        </div>
      ))}
      {isLoading && (
        <div className='flex justify-center'>
          <div className='animate-pulse bg-yellow-300 h-2 w-20 rounded'></div>
        </div>
      )}
    </div>
  );
};

export default ChatMessages;
