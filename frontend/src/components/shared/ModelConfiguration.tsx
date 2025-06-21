import React, { useState, useEffect } from 'react';
import { ChevronDown, Settings, Brain, Search, Zap, Sun, Moon, Loader2 } from 'lucide-react';

export interface ModelSettings {
  selectedModel: string;
  embeddingModel: string;
  searchType: string;
  temperature: number;
  maxTokens: number;
}

interface ModelConfigurationProps {
  settings: ModelSettings;
  onSettingsChange: (settings: Partial<ModelSettings>) => void;
  showAdvanced?: boolean;
  onToggleAdvanced?: () => void;
  theme?: 'light' | 'dark';
  onThemeChange?: (theme: 'light' | 'dark') => void;
}

export const ModelConfiguration: React.FC<ModelConfigurationProps> = ({
  settings,
  onSettingsChange,
  showAdvanced = false,
  onToggleAdvanced,
  theme = 'light',
  onThemeChange,
}) => {
  const [chatModels, setChatModels] = useState([
    { id: 'gpt-4', name: 'GPT-4', provider: 'Azure OpenAI', description: 'Most capable model' },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'Azure OpenAI', description: 'Faster and more efficient' },
    { id: 'gpt-35-turbo', name: 'GPT-3.5 Turbo', provider: 'Azure OpenAI', description: 'Fast and cost-effective' },
    { id: 'financial-llm', name: 'Financial LLM', provider: 'Industry Specific', description: 'Specialized for finance' },
    { id: 'grok-beta', name: 'Grok Beta', provider: 'xAI', description: 'Advanced reasoning' },
    { id: 'deepseek-chat', name: 'DeepSeek Chat', provider: 'DeepSeek', description: 'High performance' },
  ]);

  const [embeddingModels, setEmbeddingModels] = useState([
    { id: 'text-embedding-ada-002', name: 'Ada-002', dimensions: '1536d', provider: 'Azure OpenAI' },
    { id: 'text-embedding-3-small', name: 'Text-3-Small', dimensions: '1536d', provider: 'Azure OpenAI' },
    { id: 'text-embedding-3-large', name: 'Text-3-Large', dimensions: '3072d', provider: 'Azure OpenAI' },
  ]);

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetchFoundryModels();
  }, []);

  const fetchFoundryModels = async () => {
    setIsLoadingModels(true);
    setModelLoadError(null);
    
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/admin/foundry/models`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      
      const data = await response.json();
      const foundryModels = data.models || [];
      
      const chatModelsList: Array<{id: string, name: string, provider: string, description: string}> = [];
      const embeddingModelsList: Array<{id: string, name: string, dimensions: string, provider: string}> = [];
      
      foundryModels.forEach((model: any) => {
        if (model.type === 'chat') {
          chatModelsList.push({
            id: model.name,
            name: model.name,
            provider: 'Azure AI Foundry',
            description: `${model.version || 'Latest'} - ${model.status || 'Active'}`
          });
        } else if (model.type === 'embedding') {
          embeddingModelsList.push({
            id: model.name,
            name: model.name,
            dimensions: model.dimensions || 'Unknown',
            provider: 'Azure AI Foundry'
          });
        }
      });
      
      if (chatModelsList.length > 0) {
        setChatModels(prev => [...prev, ...chatModelsList]);
      }
      
      if (embeddingModelsList.length > 0) {
        setEmbeddingModels(prev => [...prev, ...embeddingModelsList]);
      }
      
    } catch (error) {
      console.error('Error fetching foundry models:', error);
      setModelLoadError(error instanceof Error ? error.message : 'Failed to load models');
    } finally {
      setIsLoadingModels(false);
    }
  };

  const searchTypes = [
    { id: 'hybrid', name: 'Hybrid Search', description: 'Vector + Keyword', icon: Zap },
    { id: 'vector', name: 'Vector Search', description: 'Semantic similarity', icon: Brain },
    { id: 'keyword', name: 'Keyword Search', description: 'Traditional search', icon: Search },
  ];

  const selectedChatModel = chatModels.find(m => m.id === settings.selectedModel) || chatModels[0];
  const selectedEmbeddingModel = embeddingModels.find(m => m.id === settings.embeddingModel) || embeddingModels[0];
  const selectedSearchType = searchTypes.find(s => s.id === settings.searchType) || searchTypes[0];

  return (
    <div className="bg-background border-b">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Model Configuration</h3>
            {isLoadingModels && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <div className="flex items-center gap-2">
            {onThemeChange && (
              <button
                onClick={() => onThemeChange(theme === 'light' ? 'dark' : 'light')}
                className="flex items-center gap-2 px-3 py-1 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80"
              >
                {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                {theme === 'light' ? 'Dark' : 'Light'}
              </button>
            )}
            {onToggleAdvanced && (
              <button
                onClick={onToggleAdvanced}
                className="flex items-center gap-2 px-3 py-1 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80"
              >
                <Settings className="h-4 w-4" />
                Advanced
              </button>
            )}
          </div>
        </div>

        {modelLoadError && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md dark:bg-destructive/20 dark:border-destructive/30">
            <p className="text-sm text-destructive dark:text-red-400">
              Failed to load Azure AI Foundry models: {modelLoadError}
            </p>
            <button
              onClick={fetchFoundryModels}
              className="mt-2 text-xs text-destructive hover:underline"
            >
              Retry
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Chat Model */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-muted-foreground" />
              <label className="text-sm font-medium">Chat Model</label>
            </div>
            <div className="relative">
              <select
                value={settings.selectedModel}
                onChange={(e) => onSettingsChange({ selectedModel: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
              >
                {chatModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.provider}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            </div>
            <div className="text-xs text-muted-foreground">
              Provider: {selectedChatModel.provider}
            </div>
          </div>

          {/* Embedding Model */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              <label className="text-sm font-medium">Embedding Model</label>
            </div>
            <div className="relative">
              <select
                value={settings.embeddingModel}
                onChange={(e) => onSettingsChange({ embeddingModel: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
              >
                {embeddingModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.dimensions}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            </div>
            <div className="text-xs text-muted-foreground">
              Dimensions: {selectedEmbeddingModel.dimensions}
            </div>
          </div>

          {/* Search Type */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <label className="text-sm font-medium">Search Type</label>
            </div>
            <div className="relative">
              <select
                value={settings.searchType}
                onChange={(e) => onSettingsChange({ searchType: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
              >
                {searchTypes.map((type) => (
                  <option key={type.id} value={type.id}>
                    {type.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            </div>
            <div className="text-xs text-muted-foreground">
              {selectedSearchType.description}
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        {showAdvanced && (
          <div className="mt-4 pt-4 border-t">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Temperature</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={settings.temperature}
                    onChange={(e) => onSettingsChange({ temperature: parseFloat(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm text-muted-foreground w-12">
                    {settings.temperature}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Tokens</label>
                <input
                  type="number"
                  min="100"
                  max="8000"
                  step="100"
                  value={settings.maxTokens}
                  onChange={(e) => onSettingsChange({ maxTokens: parseInt(e.target.value) })}
                  className="w-full p-2 border border-input rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
