import { useState, useEffect } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { ChatMessageWrapper } from './components/ChatMessageWrapper';
import { ArtifactsPanel } from './components/ArtifactsPanel';
import { ConfirmDialog } from './components/ConfirmDialog';
import { Layers, Trash2 } from 'lucide-react';
import { cn } from './lib/utils';
import { Message } from './types';
import axios from 'axios';
import { getTextResponse, getChatHistory } from './apiService';

interface Artifact {
  id: string;
  type: 'code' | 'text' | 'image';
  content: string;
  title: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showArtifacts, setShowArtifacts] = useState(true);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const onSendMessage = async (input: string) => {
    const tempId = `temp-${Date.now()}`;
    const newUserMessage: Message = {
      id: tempId,
      text: input,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    // Immediately update the messages state with the user's input
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setIsLoading(true);

    try {
      await getTextResponse(
        input,
        `${process.env.REACT_APP_API_BASE_URL}/chat-text/`
      );
      await getHistory(setMessages);
      setIsLoading(false);
    } catch (error) {
      console.log(error);
    } finally {
      setIsLoading(false);
    }
  };

  const getHistory = async (
    setMessages: (value: React.SetStateAction<Message[]>) => void
  ) => {
    const response = await getChatHistory(
      `${process.env.REACT_APP_API_BASE_URL}/chat_history/`
    );
    setMessages([...response]);
  };

  const handleClearHistory = async () => {
    await axios.post(
      `${process.env.REACT_APP_API_BASE_URL}/clear_chat_history/`
    );
    await getHistory(setMessages);
    setArtifacts([]);
    console.log('chat history cleared');
  };

  useEffect(() => {
    getHistory(setMessages);
  }, []);
  console.log(messages, ' message component');

  return (
    <div className='min-h-screen bg-gray-100'>
      <div className='max-w-[90rem] mx-auto min-h-screen flex'>
        {/* Main Chat Area */}
        <div
          className={cn(
            'bg-white shadow-xl flex flex-col transition-all duration-300',
            showArtifacts ? 'w-[30%]' : 'flex-1'
          )}>
          {/* Header */}
          <div className='border-b border-gray-200 p-4 flex items-center justify-between'>
            <div className='flex items-center gap-4'>
              <h1 className='text-xl font-semibold text-gray-800'>
                Sophia-Your New York Restaurant Consultant
              </h1>
              <button
                onClick={() => setShowClearConfirm(true)}
                className='p-2 hover:bg-red-50 hover:text-red-600 rounded-lg transition-colors'
                title='Clear History'>
                <Trash2 className='w-5 h-5' />
              </button>
            </div>
            <button
              onClick={() => setShowArtifacts(!showArtifacts)}
              className={cn(
                'p-2 rounded-lg transition-colors',
                showArtifacts
                  ? 'bg-blue-50 text-blue-600'
                  : 'hover:bg-gray-100 text-gray-600'
              )}>
              <Layers className='w-5 h-5' />
            </button>
          </div>

          {/* Messages Area */}
          <div>
            <ChatMessageWrapper messages={messages} isLoading={isLoading} />
          </div>

          {/* Chat Interface */}
          <ChatInterface
            onSendMessage={onSendMessage}
            setMessages={setMessages}
            getHistory={getHistory}
          />
        </div>

        {/* Artifacts Panel */}
        {showArtifacts && (
          <div className='w-[70%] bg-white border-l border-gray-200'>
            <ArtifactsPanel
              artifacts={artifacts}
              onClose={() => setShowArtifacts(false)}
            />
          </div>
        )}
      </div>

      <ConfirmDialog
        isOpen={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        onConfirm={handleClearHistory}
        title='Clear Chat History'
        message='Are you sure you want to clear all chat messages and artifacts? This action cannot be undone.'
      />
    </div>
  );
}

export default App;
