import { useState, useEffect } from 'react';

interface LoadingMessageProps {
  isLoading: boolean;
}
const thinkingMessages = [
  "Hold on a sec, foodie friend! I'm diving into my secret notebook of NYC's tastiest spots...",
  "Oh, this is exciting! I'm consulting my culinary crystal ball to find the perfect place for you...",
  "Just a sprinkle of magic, and I'll have the inside scoop! Give me a moment to check my sources...",
  'Let me peek at my notes... I’ve got all the hottest restaurants in NYC right here!',
  "Alright, let's see... I'm checking my foodie maps for the best recommendations just for you!",
];

export function LoadingMessage({ isLoading }: LoadingMessageProps) {
  const [currentThinkingMessage, setCurrentThinkingMessage] = useState(
    thinkingMessages[0]
  );
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (isLoading) {
      const intervalId = setInterval(() => {
        setMessageIndex(
          (prevIndex) => (prevIndex + 1) % thinkingMessages.length
        );
        setCurrentThinkingMessage(thinkingMessages[messageIndex]);
      }, 3000);

      return () => clearInterval(intervalId);
    }
  }, [isLoading, messageIndex]);

  if (!isLoading) {
    return null;
  }
  return (
    <div className='flex items-start mt-2 ml-4'>
      <div className='bg-gray-100 rounded-lg px-3 py-2 text-gray-700'>
        <p className='text-sm italic'>{currentThinkingMessage}</p>
        <div className='flex items-center justify-end space-x-1 mt-1'>
          <div className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'></div>
          <div
            className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'
            style={{ animationDelay: '0.2s' }}></div>
          <div
            className='w-2 h-2 bg-gray-500 rounded-full animate-bounce'
            style={{ animationDelay: '0.4s' }}></div>
        </div>
      </div>
    </div>
  );
}
