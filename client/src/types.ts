// types.ts
export interface Citation {
  source: string;
  page?: number | string | null;
  chunk_id?: string | null;
  note?: string | null;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string;
  citations?: Citation[];
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}
