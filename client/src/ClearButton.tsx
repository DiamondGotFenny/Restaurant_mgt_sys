import React from 'react';
import { FaTrash } from 'react-icons/fa';

interface ClearButtonProps {
  onClear: () => void;
}

const ClearButton: React.FC<ClearButtonProps> = ({ onClear }) => {
  return (
    <button
      onClick={onClear}
      className='flex items-center justify-center h-12 bg-red-500 text-white px-4 py-2 rounded-full hover:bg-red-600 transition duration-200'>
      <FaTrash className='mr-2' />
      Clear Chat
    </button>
  );
};

export default ClearButton;
