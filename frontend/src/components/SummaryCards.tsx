import React from 'react';
import { FileText, X, Calendar, ChevronDown, ChevronRight } from 'lucide-react';
import { useApp } from '../context/AppContext';

const SummaryCards: React.FC = () => {
  const { documents, removeDocument, selectedDocument, setSelectedDocument } = useApp();

  const handleCardClick = (docId: string) => {
    setSelectedDocument(selectedDocument === docId ? null : docId);
  };

  if (documents.length === 0) {
    return (
      <div className="bg-[#263238] rounded-lg border border-[#37474F] p-6 text-center">
        <FileText size={48} className="text-[#90A4AE] mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-[#ECEFF1] mb-2">No Documents</h3>
        <p className="text-[#90A4AE] text-sm">
          Upload PDF or TXT documents to see summaries here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-[#ECEFF1] mb-4">Document Summaries</h2>
      
      {documents.map((doc) => (
        <div
          key={doc.id}
          className={`
            bg-[#263238] rounded-lg border border-[#37474F] overflow-hidden transition-all duration-200
            ${selectedDocument === doc.id ? 'ring-2 ring-[#4FC3F7]' : 'hover:border-[#4FC3F7]'}
          `}
        >
          <div className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2 flex-1">
                <FileText size={16} className="text-[#4FC3F7] flex-shrink-0" />
                <h3 className="text-sm font-medium text-[#ECEFF1] truncate">
                  {doc.name}
                </h3>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeDocument(doc.id);
                }}
                className="p-1 hover:bg-[#37474F] rounded-lg transition-colors"
              >
                <X size={14} className="text-[#90A4AE]" />
              </button>
            </div>
            
            <div className="flex items-center gap-2 mb-3 text-xs text-[#90A4AE]">
              <Calendar size={12} />
              <span>{doc.uploadTime.toLocaleString()}</span>
              <span className="px-2 py-1 bg-[#37474F] rounded-full uppercase">
                {doc.type}
              </span>
            </div>
            
            <p className="text-sm text-[#ECEFF1] line-clamp-3 mb-3">
              {doc.summary}
            </p>
            
            <button
              onClick={() => handleCardClick(doc.id)}
              className="flex items-center gap-2 text-xs text-[#4FC3F7] hover:text-[#29B6F6] transition-colors"
            >
              {selectedDocument === doc.id ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <span>
                {selectedDocument === doc.id ? 'Collapse' : 'Expand'} Preview
              </span>
            </button>
          </div>
          
          {selectedDocument === doc.id && (
            <div className="border-t border-[#37474F] p-4 bg-[#1A2C32] animate-fadeIn">
              <div>
                <h4 className="text-sm font-medium text-[#4FC3F7] mb-2">AI-Generated Summary:</h4>
                <p className="text-sm text-[#ECEFF1] leading-relaxed mb-2">
                  {doc.summary}
              </p>
                <div className="text-xs text-[#90A4AE]">
                  Summary: {doc.summary.split(' ').length} words
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default SummaryCards;