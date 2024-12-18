import { create } from 'zustand';
import { getPromotions } from '../apiService';
import { Promotion } from '../types';

interface PromotionsState {
  promotions: Promotion[];
  cachedPromotions: Promotion[];
  isLoading: boolean;
  error: string | null;
  fetchPromotions: () => Promise<void>;
  prepareNextPromotions: () => void;
}

export const usePromotionsStore = create<PromotionsState>((set) => ({
  promotions: [],
  cachedPromotions: [],
  isLoading: false,
  error: null,

  fetchPromotions: async () => {
    try {
      set({ isLoading: true, error: null });
      const newPromotions = await getPromotions(
        `${process.env.REACT_APP_API_BASE_URL}/get_promotions/`
      );
      set({ cachedPromotions: newPromotions }); // Store in cache first
    } catch (error) {
      set({ error: 'Failed to fetch promotions' });
      console.error('Error fetching promotions:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  prepareNextPromotions: () => {
    // Move cached promotions to current promotions
    set((state) => ({
      promotions: state.cachedPromotions,
      cachedPromotions: [], // Clear cache
    }));
  },
}));
