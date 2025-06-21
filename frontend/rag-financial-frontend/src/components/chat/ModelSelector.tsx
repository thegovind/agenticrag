import React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Settings, Brain, Search, Zap } from 'lucide-react';
import { ModelSettings } from '../shared/ModelConfiguration';

interface ModelSelectorProps {
  selectedModel: string;
  embeddingModel: string;
  onModelChange: (model: string) => void;
  onEmbeddingModelChange: (model: string) => void;
  onSettingsChange: (settings: Partial<ModelSettings>) => void;
  settings: ModelSettings;
}

const CHAT_MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'Azure OpenAI', description: 'Most capable model' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'Azure OpenAI', description: 'Faster GPT-4' },
  { id: 'gpt-35-turbo', name: 'GPT-3.5 Turbo', provider: 'Azure OpenAI', description: 'Fast and efficient' },
  { id: 'financial-llm', name: 'Financial LLM', provider: 'Industry', description: 'Financial services specialized' },
  { id: 'grok', name: 'Grok', provider: 'xAI', description: 'Real-time knowledge' },
  { id: 'deepseek', name: 'DeepSeek', provider: 'DeepSeek', description: 'Code and reasoning' },
];

const EMBEDDING_MODELS = [
  { id: 'text-embedding-ada-002', name: 'Ada-002', provider: 'Azure OpenAI', dimensions: 1536 },
  { id: 'text-embedding-3-small', name: 'Embedding-3 Small', provider: 'Azure OpenAI', dimensions: 1536 },
  { id: 'text-embedding-3-large', name: 'Embedding-3 Large', provider: 'Azure OpenAI', dimensions: 3072 },
];

const SEARCH_TYPES = [
  { id: 'vector', name: 'Vector Search', icon: Brain, description: 'Semantic similarity' },
  { id: 'keyword', name: 'Keyword Search', icon: Search, description: 'Exact term matching' },
  { id: 'hybrid', name: 'Hybrid Search', icon: Zap, description: 'Vector + Keyword' },
  { id: 'semantic', name: 'Semantic Ranking', icon: Brain, description: 'AI-powered ranking' },
];

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  embeddingModel,
  onModelChange,
  onEmbeddingModelChange,
  onSettingsChange,
  settings,
}) => {

  const selectedChatModel = CHAT_MODELS.find(m => m.id === selectedModel);
  const selectedEmbeddingModel = EMBEDDING_MODELS.find(m => m.id === embeddingModel);
  const selectedSearchType = SEARCH_TYPES.find(s => s.id === settings.searchType);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Model Configuration</h3>
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4 mr-2" />
              Advanced
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80" align="end">
            <div className="space-y-4">
              <div>
                <Label htmlFor="temperature">Temperature: {settings.temperature}</Label>
                <Slider
                  id="temperature"
                  min={0}
                  max={2}
                  step={0.1}
                  value={[settings.temperature]}
                  onValueChange={([value]) => onSettingsChange({ temperature: value })}
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher values make output more random
                </p>
              </div>
              
              <div>
                <Label htmlFor="maxTokens">Max Tokens: {settings.maxTokens}</Label>
                <Slider
                  id="maxTokens"
                  min={100}
                  max={4000}
                  step={100}
                  value={[settings.maxTokens]}
                  onValueChange={([value]) => onSettingsChange({ maxTokens: value })}
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Maximum response length
                </p>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              <Label>Chat Model</Label>
            </div>
            
            <Select value={selectedModel} onValueChange={onModelChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CHAT_MODELS.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {model.provider}
                        </Badge>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {model.description}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {selectedChatModel && (
              <div className="text-xs text-muted-foreground">
                Provider: {selectedChatModel.provider}
              </div>
            )}
          </div>
        </Card>

        <Card className="p-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              <Label>Embedding Model</Label>
            </div>
            
            <Select value={embeddingModel} onValueChange={onEmbeddingModelChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EMBEDDING_MODELS.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {model.dimensions}d
                        </Badge>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {model.provider}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {selectedEmbeddingModel && (
              <div className="text-xs text-muted-foreground">
                Dimensions: {selectedEmbeddingModel.dimensions}
              </div>
            )}
          </div>
        </Card>

        <Card className="p-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              <Label>Search Type</Label>
            </div>
            
            <Select 
              value={settings.searchType} 
              onValueChange={(value: any) => onSettingsChange({ searchType: value })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SEARCH_TYPES.map((type) => (
                  <SelectItem key={type.id} value={type.id}>
                    <div className="flex items-center gap-2">
                      <type.icon className="h-4 w-4" />
                      <div className="flex flex-col">
                        <span>{type.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {type.description}
                        </span>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {selectedSearchType && (
              <div className="text-xs text-muted-foreground">
                {selectedSearchType.description}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
