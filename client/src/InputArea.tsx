import React, { useState } from 'react';
import { FaPaperPlane } from 'react-icons/fa';
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
        className='flex-grow w-64 px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500'
        placeholder='Type your message...'
      />
      <button
        type='submit'
        className='flex w-12 h-12 items-center justify-center bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition duration-200'>
        <FaPaperPlane size={20} />
      </button>
    </form>
  );
};

export default InputArea;
