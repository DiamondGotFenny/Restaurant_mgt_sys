import { Bot } from 'lucide-react';
import { cn } from '../lib/utils';

interface AiAvatarProps {
  isThinking?: boolean;
}

export function AiAvatar({ isThinking = false }: AiAvatarProps) {
  return (
    <div className='relative'>
      <div
        className={cn(
          'w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center',
          isThinking && 'animate-pulse'
        )}>
        <Bot className='w-6 h-6 text-blue-600' />
      </div>
      {isThinking && (
        <span className='absolute -bottom-6 left-1/2 -translate-x-1/2 text-sm text-gray-500'>
          Thinking...
        </span>
      )}
    </div>
  );
}
