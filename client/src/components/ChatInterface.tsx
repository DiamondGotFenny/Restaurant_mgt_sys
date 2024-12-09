import { useState } from 'react';
import { Mic, MicOff, Send } from 'lucide-react';
import { cn } from '../lib/utils';
import SpeechControl from './SpeechControl';
import { Message } from '../types';

interface ChatInterfaceProps {
  onSendMessage: (content: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  getHistory: (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => Promise<void>;
}

export function ChatInterface({
  onSendMessage,
  setMessages,
  getHistory,
}: ChatInterfaceProps) {
  const [inputMode, setInputMode] = useState<'text' | 'audio'>('text');
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    if (!inputValue.trim()) return;
    onSendMessage(inputValue.trim());
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
              placeholder='Type your message...'
              className='w-full px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500'
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            />
          ) : (
            <SpeechControl setMessages={setMessages} getHistory={getHistory} />
          )}
        </div>

        <button
          onClick={() => setInputMode(inputMode === 'text' ? 'audio' : 'text')}
          className='p-2 rounded-full hover:bg-gray-100'>
          {inputMode === 'text' ? (
            <Mic className='w-5 h-5 text-gray-600' />
          ) : (
            <MicOff className='w-5 h-5 text-gray-600' />
          )}
        </button>

        <button
          onClick={handleSend}
          disabled={!inputValue.trim() || inputMode === 'audio'}
          className={cn(
            'p-2 rounded-full text-white',
            inputValue.trim()
              ? 'bg-blue-500 hover:bg-blue-600'
              : 'bg-blue-300 cursor-not-allowed'
          )}>
          <Send className='w-5 h-5' />
        </button>
      </div>
    </div>
  );
}
