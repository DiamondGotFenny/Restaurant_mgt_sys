//useChatStore.ts
import { create } from 'zustand';
import { Message } from '../types';
import { getTextResponse, getChatHistory } from '../apiService';
import axios from 'axios';

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
  error: string | null;
  getHistory: () => Promise<void>;
  sendMessage: (input: string) => Promise<void>;
  clearHistory: () => Promise<void>;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  error: null,
  getHistory: async () => {
    try {
      set({ isLoading: true, error: null });
      const response = await getChatHistory(
        `${process.env.REACT_APP_API_BASE_URL}/chat_history/`
      );
      if (Array.isArray(response)) {
        set({ messages: response });
      } else {
        set({ error: 'Invalid response format' });
      }
    } catch (error) {
      set({ error: 'Failed to fetch chat history' });
      console.error('Error fetching history:', error);
    } finally {
      set({ isLoading: false });
    }
  },
  setLoading: (loading: boolean) => {
    set({ isLoading: loading });
  },
  sendMessage: async (input: string) => {
    try {
      set({ isLoading: true, error: null });

      // Add user message immediately
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        text: input,
        sender: 'user',
        timestamp: new Date().toISOString(),
        type: 'regular',
      };

      // Add loading message object
      const loadingMessage: Message = {
        id: `loading-${Date.now()}`,
        text: '', // This can be empty as we'll render LoadingMessage component
        sender: 'assistant',
        timestamp: new Date().toISOString(),
        type: 'loading',
      };

      // Update messages with both user message and loading message
      set((state) => ({
        messages: [...state.messages, userMessage, loadingMessage],
      }));

      // Get bot response
      const response = await getTextResponse(
        input,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text/`
      );

      // Add bot message and remove loading message
      const botMessage: Message = {
        id: response.id || `assistant-${Date.now()}`,
        text: response.text || 'Something went wrong, please try again...',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
        type: 'regular',
      };

      set((state) => ({
        messages: state.messages
          .filter((message) => message.type !== 'loading')
          .concat(botMessage),
      }));
    } catch (error) {
      set((state) => ({
        error: 'Failed to send message',
        // Remove loading message on error
        messages: state.messages.filter(
          (message) => message.type !== 'loading'
        ),
      }));
      console.error('Error sending message:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  clearHistory: async () => {
    try {
      set({ isLoading: true, error: null });
      await axios.post(
        `${process.env.REACT_APP_API_BASE_URL}/clear_chat_history/`
      );
      set({ messages: [] });
    } catch (error) {
      set({ error: 'Failed to clear history' });
      console.error('Error clearing history:', error);
    } finally {
      set({ isLoading: false });
    }
  },
}));
