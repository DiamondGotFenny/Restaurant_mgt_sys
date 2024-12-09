import { cn } from '../lib/utils';

interface ContentDisplayProps {
  content: string;
  type: 'text' | 'code' | 'image';
  className?: string;
}

export function ContentDisplay({
  content,
  type,
  className,
}: ContentDisplayProps) {
  if (type === 'code') {
    return (
      <pre
        className={cn(
          'bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto',
          className
        )}>
        <code>{content}</code>
      </pre>
    );
  }

  if (type === 'image') {
    return (
      <div className={cn('rounded-lg overflow-hidden', className)}>
        <img src={content} alt='AI Generated' className='w-full h-auto' />
      </div>
    );
  }

  return (
    <div className={cn('prose prose-blue max-w-none', className)}>
      {content}
    </div>
  );
}
