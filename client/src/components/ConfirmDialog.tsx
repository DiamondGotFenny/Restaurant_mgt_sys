import { AlertTriangle, X } from 'lucide-react';

interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
}

export function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
}: ConfirmDialogProps) {
  if (!isOpen) return null;

  return (
    <div className='fixed inset-0 bg-black/50 flex items-center justify-center z-50'>
      <div className='bg-white rounded-lg shadow-xl max-w-md w-full mx-4'>
        <div className='flex items-center justify-between p-4 border-b'>
          <div className='flex items-center gap-2'>
            <AlertTriangle className='w-5 h-5 text-yellow-500' />
            <h3 className='text-lg font-semibold'>{title}</h3>
          </div>
          <button
            onClick={onClose}
            className='p-1 hover:bg-gray-100 rounded-full'>
            <X className='w-5 h-5 text-gray-500' />
          </button>
        </div>

        <div className='p-4'>
          <p className='text-gray-600'>{message}</p>
        </div>

        <div className='flex justify-end gap-2 p-4 border-t'>
          <button
            onClick={onClose}
            className='px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg'>
            Cancel
          </button>
          <button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className='px-4 py-2 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-lg'>
            Clear History
          </button>
        </div>
      </div>
    </div>
  );
}
