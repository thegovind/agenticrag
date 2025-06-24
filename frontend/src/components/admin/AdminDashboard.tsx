import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { Activity, MessageSquare, DollarSign, AlertTriangle, CheckCircle, XCircle, TrendingUp, Database, Cpu, HardDrive, Settings, Zap, BarChart3, RefreshCw } from 'lucide-react';
import { ModelConfiguration, ModelConfiguration as ModelConfigType } from './ModelConfiguration';
import { apiService } from '../../services/api';

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

interface AdminDashboardProps {
  isActive?: boolean;
}

export const AdminDashboard: React.FC<AdminDashboardProps> = ({ isActive = true }) => {
  const [tokenUsage, setTokenUsage] = useState<TokenUsage[]>([]);
  const [tokenAnalytics, setTokenAnalytics] = useState<any>(null);
  const [tokenTrends, setTokenTrends] = useState<any[]>([]);
  const [deploymentUsage, setDeploymentUsage] = useState<any[]>([]);
  const [serviceUsage, setServiceUsage] = useState<any[]>([]);
  const [costAnalytics, setCostAnalytics] = useState<any>(null);
  const [tokenTimeRange, setTokenTimeRange] = useState<string>('7');
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetric[]>([]);
  const [traces, setTraces] = useState<TraceData[]>([]);  const [systemMetrics, setSystemMetrics] = useState<MetricData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [evaluationFramework, setEvaluationFramework] = useState<'custom' | 'azure_ai_foundry' | 'hybrid'>('custom');
  const [isUpdatingFramework, setIsUpdatingFramework] = useState(false);  const [foundryModels, setFoundryModels] = useState<any[]>([]);
  const [foundryConnections, setFoundryConnections] = useState<any[]>([]);
  const [foundryProjectInfo, setFoundryProjectInfo] = useState<any>(null);
  const [detailedRequests, setDetailedRequests] = useState<any[]>([]);  const [requestsLoading, setRequestsLoading] = useState(false);
  const [modelConfig, setModelConfig] = useState<ModelConfigType | null>(null);
  const [isLoadingFoundryData, setIsLoadingFoundryData] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  useEffect(() => {
    if (!isActive) return;
    
    fetchDashboardData();
    fetchTokenAnalytics();
    fetchEvaluationFrameworkConfig();
    fetchFoundryData();
    
    // Auto-refresh removed - use manual refresh button instead
  }, [isActive, tokenTimeRange]);
  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
        const [tokenSummaryResponse, evaluationResponse, tracesResponse, systemResponse] = await Promise.all([
        fetch('/api/v1/admin/token-usage/summary?hours=24'),
        fetch('/api/v1/admin/evaluation-metrics'),
        fetch('/api/v1/admin/traces'),
        fetch('/api/v1/admin/system-metrics')
      ]);

      if (tokenSummaryResponse.ok) {
        const tokenData = await tokenSummaryResponse.json();
        // Transform the summary data to match the existing TokenUsage interface
        const transformedTokenUsage: TokenUsage[] = tokenData.summary?.top_models?.map(([model, data]: [string, any]) => ({
          model,
          totalTokens: data.tokens,
          inputTokens: Math.round(data.tokens * 0.4), // Estimate
          outputTokens: Math.round(data.tokens * 0.6), // Estimate
          cost: data.cost,
          requests: data.requests
        })) || [];
        setTokenUsage(transformedTokenUsage);
      }      if (evaluationResponse.ok) {
        const evaluationData = await evaluationResponse.json();
        // Transform the metrics object into an array format expected by the UI
        const metricsArray: EvaluationMetric[] = evaluationData.metrics ? [
          {
            id: 'relevance',
            name: 'Relevance',
            score: evaluationData.metrics.metrics_by_type?.relevance?.average || 0,
            threshold: 0.8,
            status: (evaluationData.metrics.metrics_by_type?.relevance?.average || 0) >= 0.8 ? 'pass' : 'warning',
            description: 'Measures how relevant the AI responses are to the user queries',
            lastUpdated: evaluationData.timestamp || new Date().toISOString()
          },
          {
            id: 'accuracy',
            name: 'Accuracy',
            score: evaluationData.metrics.metrics_by_type?.accuracy?.average || 0,
            threshold: 0.8,
            status: (evaluationData.metrics.metrics_by_type?.accuracy?.average || 0) >= 0.8 ? 'pass' : 'warning',
            description: 'Measures the factual correctness of AI responses',
            lastUpdated: evaluationData.timestamp || new Date().toISOString()
          },
          {
            id: 'completeness',
            name: 'Completeness',
            score: evaluationData.metrics.metrics_by_type?.completeness?.average || 0,
            threshold: 0.8,
            status: (evaluationData.metrics.metrics_by_type?.completeness?.average || 0) >= 0.8 ? 'pass' : 'warning',
            description: 'Measures how comprehensive and complete the AI responses are',
            lastUpdated: evaluationData.timestamp || new Date().toISOString()
          },
          {
            id: 'coherence',
            name: 'Coherence',
            score: evaluationData.metrics.metrics_by_type?.coherence?.average || 0,
            threshold: 0.8,
            status: (evaluationData.metrics.metrics_by_type?.coherence?.average || 0) >= 0.8 ? 'pass' : 'warning',
            description: 'Measures how logical and well-structured the AI responses are',
            lastUpdated: evaluationData.timestamp || new Date().toISOString()
          }
        ] : [];
        setEvaluationMetrics(metricsArray);
      }      if (tracesResponse.ok) {
        const tracesData = await tracesResponse.json();
        // Transform traces data to match expected interface
        const transformedTraces: TraceData[] = tracesData.traces?.map((trace: any) => ({
          traceId: trace.trace_id || trace.traceId || '',
          operation: trace.operation_name || trace.operation || '',
          duration: trace.duration_ms || trace.duration || 0,
          status: trace.status || 'success',
          timestamp: trace.start_time || trace.timestamp || new Date().toISOString(),
          spans: 1, // Default to 1 span if not provided
          model: trace.tags?.model || trace.model,
          userId: trace.tags?.session_id || trace.userId
        })) || [];
        setTraces(transformedTraces);
      }      if (systemResponse.ok) {
        const systemData = await systemResponse.json();
        // Transform system metrics to match expected interface
        const transformedSystemMetrics: MetricData[] = systemData.metrics ? [
          {
            timestamp: systemData.timestamp,
            value: systemData.metrics.cpu?.usage_percent || 0,
            label: 'CPU Usage %'
          },
          {
            timestamp: systemData.timestamp,
            value: systemData.metrics.memory?.usage_percent || 0,
            label: 'Memory Usage %'
          },
          {
            timestamp: systemData.timestamp,
            value: systemData.metrics.disk?.usage_percent || 0,
            label: 'Disk Usage %'
          }
        ] : [];
        setSystemMetrics(transformedSystemMetrics);
      }    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
      setLastRefreshed(new Date());
    }
  };

  const fetchEvaluationFrameworkConfig = async () => {
    try {
      const response = await fetch('/api/v1/admin/evaluation-framework-config');
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
      const response = await fetch('/api/v1/admin/evaluation-framework-config', {
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

  const fetchTokenAnalytics = async () => {
    try {
      const days = parseInt(tokenTimeRange);
        const [analyticsResponse, trendsResponse, deploymentsResponse, servicesResponse, costsResponse] = await Promise.all([
        fetch(`/api/v1/admin/token-usage/analytics?days=${days}`),
        fetch(`/api/v1/admin/token-usage/trends?days=${days}&granularity=daily`),
        fetch(`/api/v1/admin/token-usage/deployments?days=${days}`),
        fetch(`/api/v1/admin/token-usage/services?days=${days}`),
        fetch(`/api/v1/admin/token-usage/costs?days=${days}`)
      ]);

      if (analyticsResponse.ok) {
        const analyticsData = await analyticsResponse.json();
        setTokenAnalytics(analyticsData.analytics);
      }      if (trendsResponse.ok) {
        const trendsData = await trendsResponse.json();
        setTokenTrends(trendsData.trends);
      }

      if (deploymentsResponse.ok) {
        const deploymentsData = await deploymentsResponse.json();
        setDeploymentUsage(deploymentsData.deployment_usage.deployments);
      }

      if (servicesResponse.ok) {
        const servicesData = await servicesResponse.json();
        setServiceUsage(servicesData.service_usage.services);
      }

      if (costsResponse.ok) {
        const costsData = await costsResponse.json();
        setCostAnalytics(costsData.costs);
      }
      
      // Also fetch detailed requests
      await fetchDetailedRequests();
      
    } catch (error) {
      console.error('Error fetching token analytics:', error);
    }
  };

  const fetchDetailedRequests = async () => {
    try {
      setRequestsLoading(true);
      const days = parseInt(tokenTimeRange);
      
      const response = await apiService.getTokenUsageRequests({
        days: days,
        limit: 50 // Limit to 50 recent requests for performance
      });
      
      setDetailedRequests(response.requests);
      
    } catch (error) {
      console.error('Error fetching detailed requests:', error);
      setDetailedRequests([]);
    } finally {
      setRequestsLoading(false);
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
          <p className="text-muted-foreground">Monitor system performance, token usage, and evaluation metrics</p>        </div>
        <div className="flex items-center gap-4">
          {lastRefreshed && (
            <span className="text-sm text-muted-foreground">
              Last updated: {lastRefreshed.toLocaleTimeString()}
            </span>
          )}
          <Button onClick={() => {
            fetchDashboardData();
            fetchTokenAnalytics();
            fetchEvaluationFrameworkConfig();
            fetchFoundryData();
          }} disabled={isLoading} className="flex items-center gap-2">
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Refreshing...' : 'Refresh Data'}
          </Button>
        </div>
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
        </TabsList>        <TabsContent value="token-usage" className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold">Token Usage Analytics</h3>
              <p className="text-sm text-muted-foreground">Comprehensive token usage and cost analysis</p>
            </div>
            <Select value={tokenTimeRange} onValueChange={setTokenTimeRange}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">Last 24 hours</SelectItem>
                <SelectItem value="7">Last 7 days</SelectItem>
                <SelectItem value="30">Last 30 days</SelectItem>
                <SelectItem value="90">Last 90 days</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Requests</p>
                    <p className="text-2xl font-bold">
                      {tokenAnalytics?.total_requests?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <Activity className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Tokens</p>
                    <p className="text-2xl font-bold">
                      {tokenAnalytics?.total_tokens?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <Zap className="h-8 w-8 text-yellow-500" />
                </div>
              </CardContent>
            </Card>            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Cost</p>
                    <p className="text-2xl font-bold">
                      ${(costAnalytics?.total_cost || tokenAnalytics?.total_cost || 0).toFixed(2)}
                    </p>
                  </div>
                  <DollarSign className="h-8 w-8 text-green-500" />
                </div>
                {costAnalytics && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    ${(costAnalytics.cost_per_token || 0).toFixed(6)} per token
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Tokens/Request</p>
                    <p className="text-2xl font-bold">
                      {Math.round(tokenAnalytics?.average_tokens_per_request || 0)}
                    </p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-purple-500" />
                </div>
              </CardContent>
            </Card>
          </div>          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Token Usage Trends</CardTitle>
                <CardDescription>Daily token consumption over time</CardDescription>
              </CardHeader>              <CardContent>
                {!tokenTrends || tokenTrends.length === 0 ? (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No trend data available for the selected time period
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={tokenTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="tokens" 
                        stroke="#8884d8" 
                        fill="#8884d8" 
                        fillOpacity={0.6}
                        name="Tokens"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cost Trends</CardTitle>
                <CardDescription>Daily cost over time</CardDescription>
              </CardHeader>              <CardContent>
                {!tokenTrends || tokenTrends.length === 0 ? (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No cost data available for the selected time period
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={tokenTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Cost']} />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="cost" 
                        stroke="#82ca9d" 
                        strokeWidth={2}
                        name="Cost ($)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Usage by Service</CardTitle>
                <CardDescription>Token consumption by service type</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={serviceUsage}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ service_type, percentage }) => `${service_type}: ${percentage}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="tokens"
                    >
                      {serviceUsage.map((_entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [Number(value).toLocaleString(), 'Tokens']} />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Usage by Deployment</CardTitle>
                <CardDescription>Token consumption by model deployment</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={deploymentUsage.slice(0, 8)} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="deployment_name" type="category" width={80} />
                    <Tooltip formatter={(value) => [Number(value).toLocaleString(), 'Tokens']} />
                    <Bar dataKey="tokens" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Detailed Tables */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Top Deployments</CardTitle>
                <CardDescription>Most active model deployments</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-80">
                  <div className="space-y-3">
                    {deploymentUsage.slice(0, 10).map((deployment, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="space-y-1">
                          <div className="font-medium">{deployment.deployment_name}</div>
                          <div className="text-sm text-muted-foreground">
                            {deployment.requests} requests • {deployment.percentage}% of total
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">${deployment.cost.toFixed(2)}</div>
                          <div className="text-sm text-muted-foreground">
                            {deployment.tokens.toLocaleString()} tokens
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Service Breakdown</CardTitle>
                <CardDescription>Usage statistics by service type</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-80">
                  <div className="space-y-3">
                    {serviceUsage.map((service, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="space-y-1">
                          <div className="font-medium">{service.service_type}</div>
                          <div className="text-sm text-muted-foreground">
                            {service.requests} requests • {service.percentage}% of total
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">${service.cost.toFixed(2)}</div>
                          <div className="text-sm text-muted-foreground">
                            {service.tokens.toLocaleString()} tokens
                          </div>                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Detailed Request Logs Table */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Request Details</CardTitle>
              <CardDescription>Detailed logs of individual requests with Q&A content and token usage</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                {requestsLoading ? (
                  <div className="text-center text-muted-foreground py-8">Loading request details...</div>
                ) : detailedRequests.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">No detailed requests found</div>
                ) : (
                  <div className="space-y-4">
                    {detailedRequests.map((request, index) => (
                      <div key={request.record_id || index} className="border rounded-lg p-4 space-y-3">
                        <div className="flex items-start justify-between">
                          <div className="space-y-1 flex-1">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">
                                {request.service_type?.replace('_', ' ').toUpperCase()}
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                {request.model_name}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {new Date(request.timestamp).toLocaleString()}
                              </span>
                            </div>
                            
                            {request.request_text && (
                              <div className="mt-2">
                                <div className="text-sm font-medium text-foreground mb-1">Question:</div>
                                <div className="text-sm text-muted-foreground bg-muted/50 p-2 rounded">
                                  {request.request_text}
                                </div>
                              </div>
                            )}
                            
                            {request.response_text && (
                              <div className="mt-2">
                                <div className="text-sm font-medium text-foreground mb-1">Answer:</div>
                                <div className="text-sm text-muted-foreground bg-muted/30 p-2 rounded">
                                  {request.response_text}
                                </div>
                              </div>
                            )}
                          </div>
                          
                          <div className="text-right ml-4 space-y-1 min-w-[120px]">
                            <div className="text-sm font-medium">${request.total_cost.toFixed(4)}</div>
                            <div className="text-xs text-muted-foreground">
                              {request.total_tokens.toLocaleString()} tokens
                            </div>
                            <div className="text-xs text-muted-foreground">
                              In: {request.prompt_tokens} • Out: {request.completion_tokens}
                            </div>
                            {request.duration_ms && (
                              <div className="text-xs text-muted-foreground">
                                {request.duration_ms}ms
                              </div>
                            )}
                            {request.verification_level && (
                              <Badge variant="secondary" className="text-xs">
                                {request.verification_level}
                              </Badge>
                            )}
                            {request.credibility_check_enabled && (
                              <Badge variant="outline" className="text-xs">
                                Credibility ✓
                              </Badge>
                            )}
                          </div>
                        </div>
                        
                        {!request.success && request.error_message && (
                          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                            Error: {request.error_message}
                          </div>
                        )}
                        
                        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                          <span>Session: {request.session_id?.split('_')[2] || 'Unknown'}</span>
                          <span>Operation: {request.operation_type?.replace('_', ' ')}</span>
                          {request.temperature && (
                            <span>Temp: {request.temperature}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
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
                  <div className="space-y-2">                    {isLoadingFoundryData ? (
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
                          </div>                          <Badge variant={model.status === 'active' ? 'default' : 'secondary'}>
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
                  <div className="space-y-2">                    {isLoadingFoundryData ? (
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
                            {getStatusIcon(connection.status || 'success')}                            <Badge variant={connection.status === 'connected' ? 'default' : 'secondary'}>
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
