import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MessageSquare, BarChart3, Database, HelpCircle, FileText } from 'lucide-react';
import { ChatContainer } from '@/components/chat/ChatContainer';
import { AdminDashboard } from '@/components/admin/AdminDashboard';
import KnowledgeBaseManager from '@/components/knowledge-base/KnowledgeBaseManager';
import { QAContainer } from '@/components/qa/QAContainer';
import SECDocumentsManager from '@/components/sec-documents/SECDocumentsManager';
import { ModelConfiguration, ModelSettings } from '@/components/shared/ModelConfiguration';
import { ThemeProvider, useTheme } from '@/contexts/ThemeContext';
import './App.css';

const AppContent = () => {
  const [activeTab, setActiveTab] = useState('sec-docs');
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [globalModelSettings, setGlobalModelSettings] = useState<ModelSettings>({
    selectedModel: 'gpt-4',
    embeddingModel: 'text-embedding-ada-002',
    searchType: 'hybrid',
    temperature: 0.7,
    maxTokens: 2000,
  });

  const { theme, setTheme } = useTheme();

  const handleModelSettingsChange = (settings: Partial<ModelSettings>) => {
    setGlobalModelSettings(prev => ({ ...prev, ...settings }));
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b">
        <div className="flex h-16 items-center px-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-6 w-6" />
              <h1 className="text-xl font-semibold">RAG Financial Assistant</h1>
            </div>
          </div>
          
          <div className="ml-auto flex items-center space-x-4">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-[1000px]">
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="chat" className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Chat
                </TabsTrigger>
                <TabsTrigger value="qa" className="flex items-center gap-2">
                  <HelpCircle className="h-4 w-4" />
                  Q&A
                </TabsTrigger>
                <TabsTrigger value="sec-docs" className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  SEC Docs
                </TabsTrigger>
                <TabsTrigger value="knowledge-base" className="flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Knowledge Base
                </TabsTrigger>
                <TabsTrigger value="admin" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Admin
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Global Model Configuration */}
      <ModelConfiguration
        settings={globalModelSettings}
        onSettingsChange={handleModelSettingsChange}
        showAdvanced={showAdvancedSettings}
        onToggleAdvanced={() => setShowAdvancedSettings(!showAdvancedSettings)}
        theme={theme}
        onThemeChange={setTheme}
      />

      <main className="flex-1">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsContent value="chat" className="m-0">
            <ChatContainer modelSettings={globalModelSettings} />
          </TabsContent>
          
          <TabsContent value="qa" className="m-0">
            <QAContainer modelSettings={globalModelSettings} />
          </TabsContent>
          
          <TabsContent value="sec-docs" className="m-0">
            <SECDocumentsManager />
          </TabsContent>
          
          <TabsContent value="knowledge-base" className="m-0">
            <KnowledgeBaseManager modelSettings={globalModelSettings} />
          </TabsContent>
          
          <TabsContent value="admin" className="m-0">
            <AdminDashboard />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
