import React, { useState } from 'react';

interface InputAreaProps {
  onSendMessage: (message: string) => void;
}

const InputArea: React.FC<InputAreaProps> = ({ onSendMessage }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className='flex space-x-2 p-4'>
      <input
        type='text'
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className='flex-grow px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
        placeholder='Type your message...'
      />
      <button
        type='submit'
        className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500'>
        Send
      </button>
    </form>
  );
};

export default InputArea;
