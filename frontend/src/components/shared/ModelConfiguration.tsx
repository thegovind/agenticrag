import React, { useState, useEffect } from 'react';
import { ChevronDown, Settings, Brain, Search, Zap, Sun, Moon, Loader2, Building2 } from 'lucide-react';

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
  theme?: 'light' | 'dark' | 'customer';
  onThemeChange?: (theme: 'light' | 'dark' | 'customer') => void;
}

export const ModelConfiguration: React.FC<ModelConfigurationProps> = ({
  settings,
  onSettingsChange,
  showAdvanced = false,
  onToggleAdvanced,
  theme = 'light',
  onThemeChange,
}) => {
  const [chatModels, setChatModels] = useState<Array<{id: string, name: string, provider: string, description: string}>>([]);

  const [embeddingModels, setEmbeddingModels] = useState<Array<{id: string, name: string, dimensions: string, provider: string}>>([]);

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [hasSetInitialDefaults, setHasSetInitialDefaults] = useState(false);

  useEffect(() => {
    fetchFoundryModels();
  }, []);

  useEffect(() => {
    // Only set defaults once when models are first loaded, don't override user selections
    if (!hasSetInitialDefaults && chatModels.length > 0 && embeddingModels.length > 0) {
      let needsUpdate = false;
      const updates: Partial<ModelSettings> = {};

      if (!settings.selectedModel) {
        // Look for preferred deployment names: chat4omini, chat4o, or chat4
        const preferredChatModel = chatModels.find(m => 
          m.id.includes('chat4omini') || m.id.includes('chat4o') || m.id.includes('chat4')
        );
        updates.selectedModel = preferredChatModel ? preferredChatModel.id : chatModels[0].id;
        needsUpdate = true;
      }

      if (!settings.embeddingModel) {
        // Look for deployment name that includes 'embedding'
        const preferredEmbeddingModel = embeddingModels.find(m => m.id.includes('embedding'));
        updates.embeddingModel = preferredEmbeddingModel ? preferredEmbeddingModel.id : embeddingModels[0].id;
        needsUpdate = true;
      }

      if (needsUpdate) {
        onSettingsChange(updates);
      }
      
      setHasSetInitialDefaults(true);
    }
  }, [chatModels, embeddingModels, settings.selectedModel, settings.embeddingModel, onSettingsChange, hasSetInitialDefaults]);

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
          // Determine dimensions based on model name
          let dimensions = 'Unknown';
          const modelName = model.model_name || model.name || '';
          if (modelName.includes('text-embedding-ada-002')) {
            dimensions = '1536d';
          } else if (modelName.includes('text-embedding-3-small')) {
            dimensions = '1536d';
          } else if (modelName.includes('text-embedding-3-large')) {
            dimensions = '3072d';
          } else if (modelName.includes('embedding')) {
            dimensions = '1536d'; // Default assumption
          }
          
          embeddingModelsList.push({
            id: model.name,
            name: model.name,
            dimensions: dimensions,
            provider: 'Azure AI Foundry'
          });
        }
      });
      
      // Set models directly (no merging since we start with empty arrays)
      if (chatModelsList.length > 0) {
        setChatModels(chatModelsList);
      }
      
      if (embeddingModelsList.length > 0) {
        setEmbeddingModels(embeddingModelsList);
      }

      // Note: Default model selection is handled by useEffect to avoid conflicts
      
    } catch (error) {
      console.error('Error fetching foundry models:', error);
      setModelLoadError(error instanceof Error ? error.message : 'Failed to load models');
      
      // Set fallback models only when API call fails
      const fallbackChatModels = [
        { id: 'gpt-4', name: 'GPT-4', provider: 'Azure OpenAI (Fallback)', description: 'Most capable model' },
        { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'Azure OpenAI (Fallback)', description: 'Faster and more efficient' },
        { id: 'gpt-35-turbo', name: 'GPT-3.5 Turbo', provider: 'Azure OpenAI (Fallback)', description: 'Fast and cost-effective' },
        { id: 'chat4o', name: 'Chat 4O', provider: 'Azure OpenAI (Fallback)', description: 'Advanced chat model' },
      ];
      
      const fallbackEmbeddingModels = [
        { id: 'text-embedding-ada-002', name: 'Ada-002', dimensions: '1536d', provider: 'Azure OpenAI (Fallback)' },
        { id: 'text-embedding-3-small', name: 'Text-3-Small', dimensions: '1536d', provider: 'Azure OpenAI (Fallback)' },
        { id: 'text-embedding-3-large', name: 'Text-3-Large', dimensions: '3072d', provider: 'Azure OpenAI (Fallback)' },
        { id: 'embedding', name: 'Embedding', dimensions: '1536d', provider: 'Azure OpenAI (Fallback)' },
      ];
      
      setChatModels(fallbackChatModels);
      setEmbeddingModels(fallbackEmbeddingModels);
      
      // Note: Default model selection is handled by useEffect to avoid conflicts
    } finally {
      setIsLoadingModels(false);
    }
  };

  const searchTypes = [
    { id: 'hybrid', name: 'Hybrid Search', description: 'Vector + Keyword', icon: Zap },
    { id: 'vector', name: 'Vector Search', description: 'Semantic similarity', icon: Brain },
    { id: 'keyword', name: 'Keyword Search', description: 'Traditional search', icon: Search },
  ];

  const selectedChatModel = chatModels.find(m => m.id === settings.selectedModel) || chatModels[0] || { id: '', name: 'Loading...', provider: '', description: '' };
  const selectedEmbeddingModel = embeddingModels.find(m => m.id === settings.embeddingModel) || embeddingModels[0] || { id: '', name: 'Loading...', dimensions: '', provider: '' };
  const selectedSearchType = searchTypes.find(s => s.id === settings.searchType) || searchTypes[0];

  return (
    <div className="border-b border-border transition-colors duration-200">
      <div className="p-4 bg-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-foreground">Model Configuration</h3>
            {isLoadingModels && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <div className="flex items-center gap-2">
            {onThemeChange && (
              <>
                <button
                  onClick={() => onThemeChange(theme === 'customer' ? 'light' : 'customer')}
                  className={`flex items-center gap-2 px-3 py-1 text-sm rounded-md transition-colors ${
                    theme === 'customer'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                  }`}
                >
                  <Building2 className="h-4 w-4" />
                  Customer
                </button>
                <button
                  onClick={() => onThemeChange(theme === 'light' ? 'dark' : 'light')}
                  className={`flex items-center gap-2 px-3 py-1 text-sm rounded-md transition-colors ${
                    theme === 'customer' ? 'opacity-50 cursor-not-allowed' : ''
                  } bg-secondary text-secondary-foreground hover:bg-secondary/80`}
                  disabled={theme === 'customer'}
                >
                  {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                  {theme === 'light' ? 'Dark' : 'Light'}
                </button>
              </>
            )}
            {onToggleAdvanced && (
              <button
                onClick={onToggleAdvanced}
                className="flex items-center gap-2 px-3 py-1 text-sm rounded-md transition-colors bg-secondary text-secondary-foreground hover:bg-secondary/80"
              >
                <Settings className="h-4 w-4" />
                Advanced
              </button>
            )}
          </div>
        </div>

        {modelLoadError && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
            <p className="text-sm text-destructive">
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
          {/* Chat Model */}            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <label className="text-sm font-medium text-foreground">Chat Model</label>
              </div>
            <div className="relative">
              <select
                value={settings.selectedModel}
                onChange={(e) => onSettingsChange({ selectedModel: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                style={{
                  colorScheme: theme === 'dark' ? 'dark' : 'light'
                }}
                disabled={chatModels.length === 0}
              >
                {chatModels.length === 0 ? (
                  <option value="" className="bg-background text-foreground">
                    Loading models...
                  </option>
                ) : (
                  chatModels.map((model) => (
                    <option 
                      key={model.id} 
                      value={model.id}
                      className="bg-background text-foreground"
                    >
                      {model.name} - {model.provider}
                    </option>
                  ))
                )}
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
              <label className="text-sm font-medium text-foreground">Embedding Model</label>
            </div>
            <div className="relative">
              <select
                value={settings.embeddingModel}
                onChange={(e) => onSettingsChange({ embeddingModel: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                style={{
                  colorScheme: theme === 'dark' ? 'dark' : 'light'
                }}
                disabled={embeddingModels.length === 0}
              >
                {embeddingModels.length === 0 ? (
                  <option value="" className="bg-background text-foreground">
                    Loading models...
                  </option>
                ) : (
                  embeddingModels.map((model) => (
                    <option 
                      key={model.id} 
                      value={model.id}
                      className="bg-background text-foreground"
                    >
                      {model.name} - {model.dimensions}
                    </option>
                  ))
                )}
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
              <label className="text-sm font-medium text-foreground">Search Type</label>
            </div>
            <div className="relative">
              <select
                value={settings.searchType}
                onChange={(e) => onSettingsChange({ searchType: e.target.value })}
                className="w-full p-2 border border-input rounded-md bg-background text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                style={{
                  colorScheme: theme === 'dark' ? 'dark' : 'light'
                }}
              >
                {searchTypes.map((type) => (
                  <option 
                    key={type.id} 
                    value={type.id}
                    className="bg-background text-foreground"
                  >
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
          <div className="mt-4 pt-4 border-t border-border">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Temperature</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={settings.temperature}
                    onChange={(e) => onSettingsChange({ temperature: parseFloat(e.target.value) })}
                    className="flex-1 accent-primary"
                    style={{
                      colorScheme: theme === 'dark' ? 'dark' : 'light'
                    }}
                  />
                  <span className="text-sm text-muted-foreground w-12">
                    {settings.temperature}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Max Tokens</label>
                <input
                  type="number"
                  min="100"
                  max="8000"
                  step="100"
                  value={settings.maxTokens}
                  onChange={(e) => onSettingsChange({ maxTokens: parseInt(e.target.value) })}
                  className="w-full p-2 border border-input rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
                  style={{
                    colorScheme: theme === 'dark' ? 'dark' : 'light'
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
