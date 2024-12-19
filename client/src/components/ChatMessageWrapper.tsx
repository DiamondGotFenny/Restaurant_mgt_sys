import { useRef, useEffect } from 'react';
import { Message } from '../types';
import { ChatMessage } from './ChatMessage';

interface ChatMessageWrapperProps {
  messages: Message[];
}

export function ChatMessageWrapper({ messages }: ChatMessageWrapperProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  return (
    <div className='flex-1 overflow-y-auto min-h-0'>
      <div className='flex flex-col'>
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            content={message.text}
            sender={message.sender}
            timestamp={message.timestamp}
            type={message.type}
          />
        ))}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
