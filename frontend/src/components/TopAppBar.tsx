import React from 'react';
import { Menu, Upload } from 'lucide-react';

interface TopAppBarProps {
  onToggleSidebar: () => void;
}

const TopAppBar: React.FC<TopAppBarProps> = ({ onToggleSidebar }) => {
  return (
    <header className="bg-[#263238] border-b border-[#37474F] px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-[#37474F] rounded-lg transition-colors lg:hidden"
        >
          <Menu size={20} />
        </button>
        
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-[#4FC3F7] rounded-lg flex items-center justify-center">
            <Upload size={16} className="text-white" />
          </div>
          <h1 className="text-xl font-bold text-[#ECEFF1]">OmniDoc AI</h1>
        </div>
      </div>
    </header>
  );
};

export default TopAppBar;