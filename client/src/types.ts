// types.ts
export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string;
  type?: 'regular' | 'loading';
}

export interface RawRoutingResult {
  is_relevant: boolean;
  vector_search_result: string | null;
  sql_result: string | null;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface Promotion {
  id: string;
  type: string;
  content: string;
  title: string;
  category?: string;
  restaurant_name?: string;
  cuisine_type?: string;
  neighborhood?: string;
}

export interface Artifact {
  id: string;
  type: 'code' | 'text' | 'image' | 'table';
  content: string;
  title: string;
}
