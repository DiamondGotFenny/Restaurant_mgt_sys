import { useRef, useEffect } from 'react';
import { Message } from '../types';
import { ChatMessage } from './ChatMessage';

interface ChatMessageWrapperProps {
  messages: Message[];
  isLoading: boolean;
}

export function ChatMessageWrapper({
  messages,
  isLoading,
}: ChatMessageWrapperProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);
  return (
    <div className='flex-1 overflow-y-auto'>
      {messages.map((message) => (
        <ChatMessage
          key={message.id}
          content={message.text}
          sender={message.sender}
          timestamp={message.timestamp}
        />
      ))}
      {isLoading && (
        <div className='flex justify-center items-center space-x-2'>
          <div className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'></div>
          <div
            className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'
            style={{ animationDelay: '0.2s' }}></div>
          <div
            className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'
            style={{ animationDelay: '0.4s' }}></div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
