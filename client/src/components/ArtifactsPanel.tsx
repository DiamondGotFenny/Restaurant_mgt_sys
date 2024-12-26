//ArtifactsPanel.tsx
import { Code2, FileText, Image, X, Table } from 'lucide-react';
import { Artifact } from '../types';
import ReactMarkdown from 'react-markdown';

interface ArtifactsPanelProps {
  artifacts: Artifact[];
  onClose: () => void;
}

export function ArtifactsPanel({ artifacts, onClose }: ArtifactsPanelProps) {
  // Helper function to render content based on artifact type
  const renderContent = (artifact: Artifact) => {
    switch (artifact.type) {
      case 'code':
        return (
          <pre className='bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto'>
            <code>{artifact.content}</code>
          </pre>
        );
      case 'image':
        return (
          <img
            src={artifact.content}
            alt={artifact.title}
            className='w-full h-auto rounded'
          />
        );
      case 'text':
        return <ReactMarkdown className='prose' children={artifact.content} />;
      case 'table':
        try {
          const data = JSON.parse(artifact.content);
          if (!Array.isArray(data) || data.length === 0) {
            return <p>No data available.</p>;
          }

          const headers = Object.keys(data[0]);

          return (
            <table className='min-w-full table-auto border-collapse border border-gray-200'>
              <thead>
                <tr>
                  {headers.map((header) => (
                    <th
                      key={header}
                      className='px-4 py-2 border border-gray-200 bg-gray-100 text-left'>
                      {header.replace(/_/g, ' ').toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row: any, index: number) => (
                  <tr
                    key={index}
                    className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {headers.map((header) => (
                      <td
                        key={header}
                        className='px-4 py-2 border border-gray-200'>
                        {row[header]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          );
        } catch (error) {
          console.error('Error parsing table content:', error);
          return <p>Error rendering table.</p>;
        }
      default:
        return <p>Unsupported artifact type.</p>;
    }
  };

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
                {artifact.type === 'table' && (
                  <Table className='w-4 h-4 text-yellow-500' />
                )}
              </div>
              <div className='p-6'>{renderContent(artifact)}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
