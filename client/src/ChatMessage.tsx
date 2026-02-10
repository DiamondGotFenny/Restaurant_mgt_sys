import React, { useRef, useEffect } from 'react';
import { Message } from './types';
import { FaUser, FaRobot } from 'react-icons/fa';

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
}

const ChatMessages: React.FC<ChatMessagesProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  return (
    <div className='flex flex-col space-y-4 p-4 h-[calc(100vh-180px)] overflow-y-auto bg-gray-100'>
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${
            message.sender === 'user' ? 'justify-end' : 'justify-start'
          }`}>
          <div
            className={`flex items-end max-w-xs lg:max-w-md xl:max-w-lg ${
              message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}>
            <div
              className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                message.sender === 'user'
                  ? 'bg-blue-500 ml-2'
                  : 'bg-gray-400 mr-2'
              }`}>
              {message.sender === 'user' ? (
                <FaUser className='text-white text-sm' />
              ) : (
                <FaRobot className='text-white text-sm' />
              )}
            </div>
            <div
              className={`px-4 py-2 rounded-lg shadow-md ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-white text-gray-800 rounded-bl-none'
              }`}>
              <div className='whitespace-pre-wrap break-words'>{message.text}</div>
              {message.sender === 'assistant' &&
                message.citations &&
                message.citations.length > 0 && (
                  <details className='mt-2 text-xs text-gray-600'>
                    <summary className='cursor-pointer select-none'>
                      Sources
                    </summary>
                    <ul className='mt-2 list-disc pl-5 space-y-1'>
                      {message.citations.map((c, idx) => (
                        <li key={`${message.id}-c-${idx}`}>
                          <div className='break-words'>
                            <span className='font-medium'>{c.source}</span>
                            {c.page !== undefined &&
                              c.page !== null &&
                              c.page !== '' && <> {`p${c.page}`}</>}
                            {c.chunk_id && <> {`#${c.chunk_id}`}</>}
                          </div>
                          {c.note && (
                            <pre className='mt-1 whitespace-pre-wrap break-words rounded bg-gray-50 p-2 text-[11px] text-gray-700'>
                              {c.note}
                            </pre>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
            </div>
          </div>
        </div>
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
};

export default ChatMessages;
