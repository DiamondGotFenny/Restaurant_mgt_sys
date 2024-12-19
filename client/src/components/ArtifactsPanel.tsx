import { Code2, FileText, Image, X } from 'lucide-react';

interface Artifact {
  id: string;
  type: 'code' | 'text' | 'image';
  content: string;
  title: string;
}

interface ArtifactsPanelProps {
  artifacts: Artifact[];
  onClose: () => void;
}

export function ArtifactsPanel({ artifacts, onClose }: ArtifactsPanelProps) {
  return (
    <div className='w-full flex flex-col'>
      {/* Header */}
      <div className='sticky top-0 bg-white border-b border-gray-200 p-4 flex items-center justify-between'>
        <h2 className='text-lg font-semibold text-gray-800 shrink-0'>
          Artifacts
        </h2>
        <button
          onClick={onClose}
          className='p-1 hover:bg-gray-100 rounded-full'>
          <X className='w-5 h-5 text-gray-500' />
        </button>
      </div>

      {/* Artifacts List */}
      <div className='flex-1 overflow-y-auto p-6 space-y-6'>
        {artifacts.length === 0 ? (
          <div className='text-center text-gray-500 py-12'>
            No artifacts generated yet
          </div>
        ) : (
          artifacts.map((artifact) => (
            <div
              key={artifact.id}
              className='border border-gray-200 rounded-lg overflow-hidden shadow-sm'>
              <div className='bg-gray-50 px-4 py-2 flex items-center gap-2'>
                {artifact.type === 'code' && (
                  <Code2 className='w-4 h-4 text-blue-500' />
                )}
                {artifact.type === 'text' && (
                  <FileText className='w-4 h-4 text-green-500' />
                )}
                {artifact.type === 'image' && (
                  <Image className='w-4 h-4 text-purple-500' />
                )}
                <span className='font-medium text-sm'>{artifact.title}</span>
              </div>
              <div className='p-6'>
                {artifact.type === 'code' ? (
                  <pre className='bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto'>
                    <code>{artifact.content}</code>
                  </pre>
                ) : artifact.type === 'image' ? (
                  <img
                    src={artifact.content}
                    alt={artifact.title}
                    className='w-full h-auto rounded'
                  />
                ) : (
                  <p className='text-gray-700 leading-relaxed'>
                    {artifact.content}
                  </p>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
