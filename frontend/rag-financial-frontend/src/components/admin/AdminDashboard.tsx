import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Activity, MessageSquare, DollarSign, AlertTriangle, CheckCircle, XCircle, TrendingUp, Database, Cpu, HardDrive } from 'lucide-react';

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

  useEffect(() => {
    fetchDashboardData();
    
    const interval = setInterval(fetchDashboardData, 30000);
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
      </div>

      <Tabs defaultValue="token-usage" className="space-y-4">
        <TabsList>
          <TabsTrigger value="token-usage">Token Usage</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation Metrics</TabsTrigger>
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
