import { Promotion } from '../types';
import { X } from 'lucide-react';

interface PromotionsPanelProps {
  promotions: Promotion[];
  onClose: () => void;
}

export function PromotionsPanel({ promotions, onClose }: PromotionsPanelProps) {
  return (
    <div className='w-full flex flex-col'>
      {/* Header */}
      <div className='sticky top-0 bg-white border-b border-gray-200 p-4 flex items-center justify-between shrink-0'>
        <h2 className='text-lg font-semibold text-gray-800'>
          Special Recommendations
        </h2>
        <button
          onClick={onClose}
          className='p-1 hover:bg-gray-100 rounded-full'>
          <X className='w-5 h-5 text-gray-500' />
        </button>
      </div>

      {/* Promotions Content */}
      <div className='flex-1 overflow-y-auto p-6 space-y-6 m-4'>
        <div className='space-y-4'>
          <h3 className='text-lg font-medium text-gray-700'>
            While Sophia is working on your query, you can check out these
            dining recommendations:
          </h3>
          {promotions.map((promo) => (
            <div
              key={promo.id}
              className='border border-gray-200 rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow'>
              <div className='flex items-center justify-between mb-2'>
                <h4 className='font-medium text-blue-600'>{promo.title}</h4>
                {promo.cuisine_type && (
                  <span className='text-sm px-2 py-1 bg-blue-50 text-blue-600 rounded-full'>
                    {promo.cuisine_type}
                  </span>
                )}
              </div>
              <p className='text-gray-600'>{promo.content}</p>
              {promo.neighborhood && (
                <p className='mt-2 text-sm text-gray-500'>
                  📍 {promo.neighborhood}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
