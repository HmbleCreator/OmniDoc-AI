import React, { useRef } from 'react';
import { Upload, FileText } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';

const FileUpload: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addDocument, setIsUploading, apiKeys, selectedProvider } = useApp();

  const validateSetup = (): { isValid: boolean; message: string } => {
    // Check if any LLM provider is selected
    if (!selectedProvider) {
      return {
        isValid: false,
        message: 'Please select an LLM provider first before uploading files.'
      };
    }

    // Check if API key is provided for the selected provider
    const selectedApiKey = apiKeys[selectedProvider];
    if (!selectedApiKey || selectedApiKey.trim() === '') {
      return {
        isValid: false,
        message: 'API key is required. Please select your LLM provider and enter your API key in the sidebar before uploading files.'
      };
    }

    return { isValid: true, message: '' };
  };

  const handleFileSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    // Validate setup before proceeding
    const validation = validateSetup();
    if (!validation.isValid) {
      alert(validation.message);
      return;
    }

    const file = files[0];
    if (file.type !== 'application/pdf' && file.type !== 'text/plain') {
      alert('Please upload only PDF or TXT files');
      return;
    }

    setIsUploading(true);

    try {
      // Upload file to backend with API keys
      const uploadedDoc = await apiService.uploadDocument(file, apiKeys, selectedProvider);
      
      // Convert backend response to frontend format
      const doc = {
        id: uploadedDoc.id,
        name: uploadedDoc.name,
        type: uploadedDoc.type as 'pdf' | 'txt',
        content: uploadedDoc.content,
        summary: uploadedDoc.summary,
        uploadTime: new Date(uploadedDoc.upload_time)
      };

      addDocument(doc);
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleUploadClick = () => {
    // Validate setup before opening file dialog
    const validation = validateSetup();
    if (!validation.isValid) {
      alert(validation.message);
      return;
    }
    
    fileInputRef.current?.click();
  };

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => handleFileSelect(e.target.files)}
        className="hidden"
      />
      
      <button
        onClick={handleUploadClick}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors ${
          validateSetup().isValid
            ? 'bg-[#37474F] text-[#ECEFF1] border-[#455A64] hover:bg-[#455A64]'
            : 'bg-yellow-900/20 text-yellow-400 border-yellow-600/50 hover:bg-yellow-900/30 cursor-not-allowed'
        }`}
        title={validateSetup().isValid ? "Upload document" : validateSetup().message}
      >
        <Upload size={16} />
        <span className="text-sm">Upload</span>
      </button>
    </>
  );
};

export default FileUpload;