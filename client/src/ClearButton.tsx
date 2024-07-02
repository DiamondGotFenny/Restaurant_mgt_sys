import React from 'react';
import { FaTrash } from 'react-icons/fa';

interface ClearButtonProps {
  onClear: () => void;
}

const ClearButton: React.FC<ClearButtonProps> = ({ onClear }) => {
  return (
    <button
      onClick={onClear}
      className='flex items-center justify-center px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500'>
      <FaTrash className='mr-2' /> Clear Chat
    </button>
  );
};

export default ClearButton;
