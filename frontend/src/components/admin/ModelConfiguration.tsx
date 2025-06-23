import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, RefreshCw, Settings, Cpu, Database, AlertTriangle, CheckCircle } from 'lucide-react';
import { deploymentService, DeploymentSummary } from '@/services/deploymentService';

interface ModelConfigurationProps {
  onConfigurationChange?: (config: ModelConfiguration) => void;
  className?: string;
}

export interface ModelConfiguration {
  chatModel: string;
  embeddingModel: string;
  chatDeployment: string;
  embeddingDeployment: string;
}

export const ModelConfiguration: React.FC<ModelConfigurationProps> = ({
  onConfigurationChange,
  className = ""
}) => {
  const [deployments, setDeployments] = useState<DeploymentSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedChatDeployment, setSelectedChatDeployment] = useState<string>('');
  const [selectedEmbeddingDeployment, setSelectedEmbeddingDeployment] = useState<string>('');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDeployments();
  }, []);

  useEffect(() => {
    // Notify parent of configuration changes
    if (selectedChatDeployment && selectedEmbeddingDeployment && deployments) {
      const chatDeployment = deployments.chat_models.find(d => d.deployment_name === selectedChatDeployment);
      const embeddingDeployment = deployments.embedding_models.find(d => d.deployment_name === selectedEmbeddingDeployment);
      
      if (chatDeployment && embeddingDeployment) {
        const config: ModelConfiguration = {
          chatModel: chatDeployment.model_name,
          embeddingModel: embeddingDeployment.model_name,
          chatDeployment: chatDeployment.deployment_name,
          embeddingDeployment: embeddingDeployment.deployment_name
        };
        onConfigurationChange?.(config);
      }
    }
  }, [selectedChatDeployment, selectedEmbeddingDeployment, deployments, onConfigurationChange]);

  const loadDeployments = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await deploymentService.getAllDeployments();
      setDeployments(data);
      
      // Auto-select first available deployments
      if (data.chat_models.length > 0 && !selectedChatDeployment) {
        setSelectedChatDeployment(data.chat_models[0].deployment_name);
      }
      if (data.embedding_models.length > 0 && !selectedEmbeddingDeployment) {
        setSelectedEmbeddingDeployment(data.embedding_models[0].deployment_name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load deployments');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDeployments();
    setRefreshing(false);
  };

  const getDeploymentStatusBadge = (state: string) => {
    const isSucceeded = state === 'Succeeded';
    return (
      <Badge variant={isSucceeded ? "default" : "destructive"} className="text-xs">
        {isSucceeded ? <CheckCircle className="h-3 w-3 mr-1" /> : <AlertTriangle className="h-3 w-3 mr-1" />}
        {state}
      </Badge>
    );
  };

  const formatCapacity = (capacity: number) => {
    if (capacity >= 1000) {
      return `${(capacity / 1000).toFixed(1)}K`;
    }
    return capacity.toString();
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Model Configuration</span>
          </CardTitle>
          <CardDescription>Configure Azure OpenAI model deployments</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span>Loading deployments...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Model Configuration</span>
          </CardTitle>
          <CardDescription>Configure Azure OpenAI model deployments</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {error}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleRefresh}
                className="ml-3"
              >
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Settings className="h-5 w-5" />
              <span>Model Configuration</span>
            </CardTitle>
            <CardDescription>
              Configure Azure OpenAI model deployments from {deployments?.account_name}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Account Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg">
          <div className="text-sm">
            <span className="font-medium">Account:</span>
            <p className="text-muted-foreground">{deployments?.account_name}</p>
          </div>
          <div className="text-sm">
            <span className="font-medium">Resource Group:</span>
            <p className="text-muted-foreground">{deployments?.resource_group}</p>
          </div>
          <div className="text-sm">
            <span className="font-medium">Total Deployments:</span>
            <p className="text-muted-foreground">{deployments?.total_deployments}</p>
          </div>
        </div>

        {/* Chat Model Selection */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Cpu className="h-4 w-4" />
            <h3 className="font-medium">Chat Model</h3>
          </div>
          <Select value={selectedChatDeployment} onValueChange={setSelectedChatDeployment}>
            <SelectTrigger>
              <SelectValue placeholder="Select a chat model deployment" />
            </SelectTrigger>
            <SelectContent>
              {deployments?.chat_models.map((deployment) => (
                <SelectItem key={deployment.deployment_name} value={deployment.deployment_name}>
                  <div className="flex items-center justify-between w-full">
                    <div className="flex flex-col">
                      <span className="font-medium">{deployment.model_name}</span>
                      <span className="text-xs text-muted-foreground">
                        {deployment.deployment_name} • {deployment.sku} • {formatCapacity(deployment.capacity)} tokens
                      </span>
                    </div>
                    <div className="ml-2">
                      {getDeploymentStatusBadge(deployment.state)}
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedChatDeployment && deployments && (
            <div className="text-xs text-muted-foreground p-2 bg-muted/30 rounded">
              <strong>Selected:</strong> {deployments.chat_models.find(d => d.deployment_name === selectedChatDeployment)?.display_name}
            </div>
          )}
        </div>

        {/* Embedding Model Selection */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Database className="h-4 w-4" />
            <h3 className="font-medium">Embedding Model</h3>
          </div>
          <Select value={selectedEmbeddingDeployment} onValueChange={setSelectedEmbeddingDeployment}>
            <SelectTrigger>
              <SelectValue placeholder="Select an embedding model deployment" />
            </SelectTrigger>
            <SelectContent>
              {deployments?.embedding_models.map((deployment) => (
                <SelectItem key={deployment.deployment_name} value={deployment.deployment_name}>
                  <div className="flex items-center justify-between w-full">
                    <div className="flex flex-col">
                      <span className="font-medium">{deployment.model_name}</span>
                      <span className="text-xs text-muted-foreground">
                        {deployment.deployment_name} • {deployment.sku} • {formatCapacity(deployment.capacity)} tokens
                      </span>
                    </div>
                    <div className="ml-2">
                      {getDeploymentStatusBadge(deployment.state)}
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedEmbeddingDeployment && deployments && (
            <div className="text-xs text-muted-foreground p-2 bg-muted/30 rounded">
              <strong>Selected:</strong> {deployments.embedding_models.find(d => d.deployment_name === selectedEmbeddingDeployment)?.display_name}
            </div>
          )}
        </div>

        {/* Summary */}
        {selectedChatDeployment && selectedEmbeddingDeployment && (
          <div className="p-4 border rounded-lg bg-green-50 dark:bg-green-900/20">
            <h4 className="font-medium text-green-900 dark:text-green-100 mb-2">Configuration Ready</h4>
            <div className="text-sm text-green-800 dark:text-green-200 space-y-1">
              <p><strong>Chat:</strong> {deployments?.chat_models.find(d => d.deployment_name === selectedChatDeployment)?.model_name}</p>
              <p><strong>Embedding:</strong> {deployments?.embedding_models.find(d => d.deployment_name === selectedEmbeddingDeployment)?.model_name}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
