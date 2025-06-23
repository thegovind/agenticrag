import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Activity, MessageSquare, DollarSign, AlertTriangle, CheckCircle, XCircle, TrendingUp, Database, Cpu, HardDrive, Settings } from 'lucide-react';
import { ModelConfiguration, ModelConfiguration as ModelConfigType } from './ModelConfiguration';

interface MetricData {
  timestamp: string;
  value: number;
  label?: string;
}

interface TokenUsage {
  model: string;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
  requests: number;
}

interface EvaluationMetric {
  id: string;
  name: string;
  score: number;
  threshold: number;
  status: 'pass' | 'fail' | 'warning';
  description: string;
  lastUpdated: string;
}

interface TraceData {
  traceId: string;
  operation: string;
  duration: number;
  status: 'success' | 'error' | 'timeout';
  timestamp: string;
  spans: number;
  model?: string;
  userId?: string;
}

export const AdminDashboard: React.FC = () => {
  const [tokenUsage, setTokenUsage] = useState<TokenUsage[]>([]);
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetric[]>([]);
  const [traces, setTraces] = useState<TraceData[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<MetricData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [evaluationFramework, setEvaluationFramework] = useState<'custom' | 'azure_ai_foundry' | 'hybrid'>('custom');
  const [isUpdatingFramework, setIsUpdatingFramework] = useState(false);
  const [foundryModels, setFoundryModels] = useState<any[]>([]);
  const [foundryConnections, setFoundryConnections] = useState<any[]>([]);
  const [foundryProjectInfo, setFoundryProjectInfo] = useState<any>(null);
  const [modelConfig, setModelConfig] = useState<ModelConfigType | null>(null);
  const [isLoadingFoundryData, setIsLoadingFoundryData] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    fetchEvaluationFrameworkConfig();
    fetchFoundryData();
    
    const interval = setInterval(() => {
      fetchDashboardData();
      fetchFoundryData();
    }, 30000);
    setRefreshInterval(interval);
    
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      
      const [tokenResponse, metricsResponse, tracesResponse, systemResponse] = await Promise.all([
        fetch('/api/admin/token-usage'),
        fetch('/api/admin/evaluation-metrics'),
        fetch('/api/admin/traces'),
        fetch('/api/admin/system-metrics')
      ]);

      if (tokenResponse.ok) {
        const tokenData = await tokenResponse.json();
        setTokenUsage(tokenData);
      }

      if (metricsResponse.ok) {
        const metricsData = await metricsResponse.json();
        setEvaluationMetrics(metricsData);
      }

      if (tracesResponse.ok) {
        const tracesData = await tracesResponse.json();
        setTraces(tracesData);
      }

      if (systemResponse.ok) {
        const systemData = await systemResponse.json();
        setSystemMetrics(systemData);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchEvaluationFrameworkConfig = async () => {
    try {
      const response = await fetch('/api/admin/evaluation-framework-config');
      if (response.ok) {
        const config = await response.json();
        setEvaluationFramework(config.framework_type || 'custom');
      }
    } catch (error) {
      console.error('Error fetching evaluation framework config:', error);
    }
  };

  const updateEvaluationFramework = async (framework: 'custom' | 'azure_ai_foundry' | 'hybrid') => {
    try {
      setIsUpdatingFramework(true);
      const response = await fetch('/api/admin/evaluation-framework-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ framework_type: framework }),
      });

      if (response.ok) {
        setEvaluationFramework(framework);
        fetchDashboardData();
      } else {
        console.error('Failed to update evaluation framework');
      }
    } catch (error) {
      console.error('Error updating evaluation framework:', error);
    } finally {
      setIsUpdatingFramework(false);
    }
  };

  const fetchFoundryData = async () => {
    try {
      setIsLoadingFoundryData(true);
      
      const [modelsResponse, connectionsResponse, projectResponse] = await Promise.all([
        fetch('/api/v1/admin/foundry/models'),
        fetch('/api/v1/admin/foundry/connections'),
        fetch('/api/v1/admin/foundry/project-info')
      ]);

      if (modelsResponse.ok) {
        const modelsData = await modelsResponse.json();
        setFoundryModels(modelsData.models || []);
      }

      if (connectionsResponse.ok) {
        const connectionsData = await connectionsResponse.json();
        setFoundryConnections(connectionsData.connections || []);
      }

      if (projectResponse.ok) {
        const projectData = await projectResponse.json();
        setFoundryProjectInfo(projectData);
      }
    } catch (error) {
      console.error('Error fetching foundry data:', error);
    } finally {
      setIsLoadingFoundryData(false);
    }
  };

  const getTotalTokens = () => tokenUsage.reduce((sum, usage) => sum + usage.totalTokens, 0);
  const getTotalCost = () => tokenUsage.reduce((sum, usage) => sum + usage.cost, 0);
  const getTotalRequests = () => tokenUsage.reduce((sum, usage) => sum + usage.requests, 0);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass':
      case 'success':
        return 'text-green-600';
      case 'fail':
      case 'error':
        return 'text-red-600';
      case 'warning':
      case 'timeout':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
      case 'success':
        return <CheckCircle className="h-4 w-4" />;
      case 'fail':
      case 'error':
        return <XCircle className="h-4 w-4" />;
      case 'warning':
      case 'timeout':
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Admin Dashboard</h1>
          <p className="text-muted-foreground">Monitor system performance, token usage, and evaluation metrics</p>
        </div>
        <Button onClick={fetchDashboardData} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Refresh Data'}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{getTotalTokens().toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Across all models</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${getTotalCost().toFixed(2)}</div>
            <p className="text-xs text-muted-foreground">Current billing period</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{getTotalRequests().toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Last 24 hours</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">Healthy</div>
            <p className="text-xs text-muted-foreground">All systems operational</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Configuration</CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-sm space-y-1">
              {modelConfig ? (
                <>
                  <div><strong>Chat:</strong> {modelConfig.chatModel}</div>
                  <div><strong>Embedding:</strong> {modelConfig.embeddingModel}</div>
                </>
              ) : (
                <div className="text-muted-foreground">Not configured</div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="token-usage" className="space-y-4">
        <TabsList>
          <TabsTrigger value="token-usage">Token Usage</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation Metrics</TabsTrigger>
          <TabsTrigger value="model-config">Model Configuration</TabsTrigger>
          <TabsTrigger value="foundry">Azure AI Foundry</TabsTrigger>
          <TabsTrigger value="tracing">Distributed Tracing</TabsTrigger>
          <TabsTrigger value="system">System Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="token-usage" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Token Usage by Model</CardTitle>
                <CardDescription>Breakdown of token consumption across different models</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={tokenUsage}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="inputTokens" stackId="a" fill="#8884d8" name="Input Tokens" />
                    <Bar dataKey="outputTokens" stackId="a" fill="#82ca9d" name="Output Tokens" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cost Distribution</CardTitle>
                <CardDescription>Cost breakdown by model</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={tokenUsage}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ model, cost }) => `${model}: $${cost.toFixed(2)}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="cost"
                    >
                      {tokenUsage.map((_entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Detailed Token Usage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {tokenUsage.map((usage, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="space-y-1">
                      <div className="font-medium">{usage.model}</div>
                      <div className="text-sm text-muted-foreground">
                        {usage.requests} requests • {usage.totalTokens.toLocaleString()} tokens
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">${usage.cost.toFixed(2)}</div>
                      <div className="text-sm text-muted-foreground">
                        {usage.inputTokens.toLocaleString()} in • {usage.outputTokens.toLocaleString()} out
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evaluation" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Evaluation Framework Configuration
                  </CardTitle>
                  <CardDescription>
                    Choose the evaluation framework for assessing RAG and agent performance
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Select
                    value={evaluationFramework}
                    onValueChange={(value: 'custom' | 'azure_ai_foundry' | 'hybrid') => updateEvaluationFramework(value)}
                    disabled={isUpdatingFramework}
                  >
                    <SelectTrigger className="w-48">
                      <SelectValue placeholder="Select framework" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="custom">
                        <div className="flex flex-col">
                          <span className="font-medium">Custom Evaluators</span>
                          <span className="text-xs text-muted-foreground">Built-in evaluation metrics</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="azure_ai_foundry">
                        <div className="flex flex-col">
                          <span className="font-medium">Azure AI Foundry</span>
                          <span className="text-xs text-muted-foreground">RAG &amp; Agent evaluators</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="hybrid">
                        <div className="flex flex-col">
                          <span className="font-medium">Hybrid Approach</span>
                          <span className="text-xs text-muted-foreground">Both custom and Azure AI Foundry</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  {isUpdatingFramework && (
                    <div className="text-sm text-muted-foreground">Updating...</div>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Current Framework:</span>
                  <Badge variant={evaluationFramework === 'custom' ? 'default' : evaluationFramework === 'azure_ai_foundry' ? 'secondary' : 'outline'}>
                    {evaluationFramework === 'custom' ? 'Custom' : evaluationFramework === 'azure_ai_foundry' ? 'Azure AI Foundry' : 'Hybrid'}
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  {evaluationFramework === 'custom' && 'Using built-in evaluation metrics for relevance, groundedness, coherence, and financial accuracy.'}
                  {evaluationFramework === 'azure_ai_foundry' && 'Using Azure AI Foundry RAG and Agent evaluators for comprehensive assessment.'}
                  {evaluationFramework === 'hybrid' && 'Combining both custom and Azure AI Foundry evaluators for maximum coverage.'}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {evaluationMetrics.map((metric) => (
              <Card key={metric.id}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{metric.name}</CardTitle>
                    <div className={`flex items-center gap-1 ${getStatusColor(metric.status)}`}>
                      {getStatusIcon(metric.status)}
                      <Badge variant={metric.status === 'pass' ? 'default' : metric.status === 'warning' ? 'secondary' : 'destructive'}>
                        {metric.status}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Score</span>
                      <span className="font-medium">{(metric.score * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={metric.score * 100} className="h-2" />
                    <div className="flex justify-between text-sm">
                      <span>Threshold</span>
                      <span>{(metric.threshold * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-xs text-muted-foreground">{metric.description}</p>
                    <p className="text-xs text-muted-foreground">
                      Last updated: {new Date(metric.lastUpdated).toLocaleString()}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="model-config" className="space-y-4">
          <ModelConfiguration 
            onConfigurationChange={setModelConfig}
            className="max-w-4xl"
          />
        </TabsContent>

        <TabsContent value="foundry" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Available Models</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{foundryModels.length}</div>
                <p className="text-xs text-muted-foreground">
                  {foundryModels.filter(m => m.type === 'chat').length} chat, {foundryModels.filter(m => m.type === 'embedding').length} embedding
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Project Connections</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{foundryConnections.length}</div>
                <p className="text-xs text-muted-foreground">Active connections</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Project Status</CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">
                  {foundryProjectInfo?.status || 'Active'}
                </div>
                <p className="text-xs text-muted-foreground">
                  {foundryProjectInfo?.name || 'RAG Financial Project'}
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Available Models</CardTitle>
                <CardDescription>Models deployed in Azure AI Foundry project</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {isLoadingFoundryData ? (
                      <div className="text-center text-muted-foreground">Loading models...</div>
                    ) : foundryModels.length === 0 ? (
                      <div className="text-center text-muted-foreground">No models found</div>
                    ) : (
                      foundryModels.map((model, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="space-y-1">
                            <div className="font-medium">{model.name}</div>
                            <div className="text-sm text-muted-foreground">
                              Type: {model.type} • Version: {model.version || 'Latest'}
                            </div>
                          </div>
                          <Badge variant={model.status === 'active' ? 'default' : 'secondary'}>
                            {model.status || 'Active'}
                          </Badge>
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Project Connections</CardTitle>
                <CardDescription>External service connections</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {isLoadingFoundryData ? (
                      <div className="text-center text-muted-foreground">Loading connections...</div>
                    ) : foundryConnections.length === 0 ? (
                      <div className="text-center text-muted-foreground">No connections found</div>
                    ) : (
                      foundryConnections.map((connection, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="space-y-1">
                            <div className="font-medium">{connection.name}</div>
                            <div className="text-sm text-muted-foreground">
                              Type: {connection.type} • {connection.description || 'External service'}
                            </div>
                          </div>
                          <div className={`flex items-center gap-1 ${getStatusColor(connection.status || 'success')}`}>
                            {getStatusIcon(connection.status || 'success')}
                            <Badge variant={connection.status === 'connected' ? 'default' : 'secondary'}>
                              {connection.status || 'Connected'}
                            </Badge>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {foundryProjectInfo && (
            <Card>
              <CardHeader>
                <CardTitle>Project Information</CardTitle>
                <CardDescription>Azure AI Foundry project details</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Project Name</div>
                    <div className="text-sm text-muted-foreground">{foundryProjectInfo.name || 'N/A'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Resource Group</div>
                    <div className="text-sm text-muted-foreground">{foundryProjectInfo.resource_group || 'N/A'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Location</div>
                    <div className="text-sm text-muted-foreground">{foundryProjectInfo.location || 'N/A'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Subscription</div>
                    <div className="text-sm text-muted-foreground">{foundryProjectInfo.subscription_id || 'N/A'}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="tracing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Traces</CardTitle>
              <CardDescription>Distributed tracing data for system operations</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-2">
                  {traces.map((trace) => (
                    <div key={trace.traceId} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{trace.operation}</span>
                          <div className={`flex items-center gap-1 ${getStatusColor(trace.status)}`}>
                            {getStatusIcon(trace.status)}
                            <Badge variant={trace.status === 'success' ? 'default' : 'destructive'}>
                              {trace.status}
                            </Badge>
                          </div>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Trace ID: {trace.traceId} • {trace.spans} spans
                          {trace.model && ` • ${trace.model}`}
                          {trace.userId && ` • User: ${trace.userId}`}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{trace.duration}ms</div>
                        <div className="text-sm text-muted-foreground">
                          {new Date(trace.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>System Performance</CardTitle>
                <CardDescription>Real-time system metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={systemMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Resource Utilization</CardTitle>
                <CardDescription>Current system resource usage</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4" />
                        <span className="text-sm">CPU Usage</span>
                      </div>
                      <span className="text-sm font-medium">65%</span>
                    </div>
                    <Progress value={65} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <HardDrive className="h-4 w-4" />
                        <span className="text-sm">Memory Usage</span>
                      </div>
                      <span className="text-sm font-medium">78%</span>
                    </div>
                    <Progress value={78} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Database className="h-4 w-4" />
                        <span className="text-sm">Storage Usage</span>
                      </div>
                      <span className="text-sm font-medium">45%</span>
                    </div>
                    <Progress value={45} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
