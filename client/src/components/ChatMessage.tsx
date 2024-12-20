import { AiAvatar } from './AiAvatar';
import { ContentDisplay } from './ContentDisplay';
import { LoadingMessage } from './LoadingMessage';
import { cn } from '../lib/utils';
interface ChatMessageProps {
  content: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string;
  contentType?: 'text' | 'code' | 'image';
  type?: 'regular' | 'loading';
}

export function ChatMessage({
  content,
  sender,
  timestamp,
  type = 'regular',
  contentType = 'text',
}: ChatMessageProps) {
  return (
    <div
      className={cn(
        'flex gap-4 p-4',
        sender === 'assistant' ? 'bg-gray-50' : 'bg-white'
      )}>
      {sender === 'assistant' && <AiAvatar />}

      <div className='flex-1'>
        <div className='flex items-center gap-2 mb-2'>
          <span className='font-medium'>
            {sender === 'assistant' ? 'Sophia' : 'user'}
          </span>
          <span className='text-sm text-gray-500'>{timestamp}</span>
        </div>

        {type === 'loading' ? (
          <LoadingMessage isLoading={true} />
        ) : typeof content === 'string' ? (
          <ContentDisplay
            content={content}
            type={contentType}
            className='min-w-0'
          />
        ) : (
          content
        )}
      </div>
    </div>
  );
}
