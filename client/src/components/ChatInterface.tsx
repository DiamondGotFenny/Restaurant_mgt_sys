import { useState } from 'react';
import { Mic, MicOff, Send } from 'lucide-react';
import { cn } from '../lib/utils';
import SpeechControl from './SpeechControl';
import { useChatStore } from '../store/useChatStore';

export function ChatInterface() {
  const { sendMessage, isLoading } = useChatStore();
  const [inputMode, setInputMode] = useState<'text' | 'audio'>('text');
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    if (!inputValue.trim() || isLoading) return;
    sendMessage(inputValue.trim());
    setInputValue('');
  };

  return (
    <div className='border-t border-gray-200 p-4'>
      <div className='flex items-center gap-2'>
        <div className='flex-1'>
          {inputMode === 'text' ? (
            <input
              type='text'
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                isLoading ? 'Waiting for response...' : 'Type your message...'
              }
              className={cn(
                'w-full px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500',
                isLoading && 'bg-gray-100 cursor-not-allowed'
              )}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              disabled={isLoading}
            />
          ) : (
            <SpeechControl />
          )}
        </div>

        <button
          onClick={() => setInputMode(inputMode === 'text' ? 'audio' : 'text')}
          className={cn(
            'p-2 rounded-full',
            isLoading ? 'text-gray-400 cursor-not-allowed' : 'hover:bg-gray-100'
          )}
          disabled={isLoading}>
          {inputMode === 'text' ? (
            <Mic className='w-5 h-5 text-gray-600' />
          ) : (
            <MicOff className='w-5 h-5 text-gray-600' />
          )}
        </button>

        <button
          onClick={handleSend}
          disabled={!inputValue.trim() || inputMode === 'audio' || isLoading}
          className={cn(
            'p-2 rounded-full text-white',
            inputValue.trim() && !isLoading
              ? 'bg-blue-500 hover:bg-blue-600'
              : 'bg-blue-300 cursor-not-allowed'
          )}>
          <Send className='w-5 h-5' />
        </button>
      </div>
    </div>
  );
}
