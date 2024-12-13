// types.ts
export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string;
  type?: 'regular' | 'loading';
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}
