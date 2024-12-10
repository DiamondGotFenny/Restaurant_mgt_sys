import { useState, useEffect } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { ChatMessageWrapper } from './components/ChatMessageWrapper';
import { ArtifactsPanel } from './components/ArtifactsPanel';
import { ConfirmDialog } from './components/ConfirmDialog';
import { Layers, Trash2 } from 'lucide-react';
import { cn } from './lib/utils';

import { useChatStore } from './store/useChatStore';

interface Artifact {
  id: string;
  type: 'code' | 'text' | 'image';
  content: string;
  title: string;
}

function App() {
  const { messages, isLoading, getHistory, clearHistory } = useChatStore();

  const [showArtifacts, setShowArtifacts] = useState(true);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  useEffect(() => {
    getHistory();
  }, []);
  console.log(messages, ' message component');

  // Debug messages changes
  useEffect(() => {
    console.log('Messages updated:', messages);
  }, [messages]);

  const handleClearHistory = async () => {
    await clearHistory();
    setArtifacts([]);
    setShowClearConfirm(false);
    getHistory();
  };
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
          <ChatInterface />
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
