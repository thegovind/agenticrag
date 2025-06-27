import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, AreaChart, Area
} from 'recharts';
import { 
  Users, FileText, Target, Activity, AlertCircle
} from 'lucide-react';
import { apiService } from '../../services/api';

interface EvaluationAnalytics {
  total_evaluations: number;
  average_scores: {
    overall: number;
    groundedness: number;
    relevance: number;
    coherence: number;
    fluency: number;
  };
  evaluator_distribution: {
    foundry: number;
    custom: number;
  };
  rag_method_performance: Array<{
    method: string;
    count: number;
    avg_score: number;
  }>;
  daily_trends: Array<{
    date: string;
    evaluations: number;
    avg_score: number;
  }>;
  score_distribution: Array<{
    range: string;
    count: number;
  }>;
  top_performing_sessions: Array<{
    session_id: string;
    avg_score: number;
    evaluation_count: number;
  }>;
}

interface SessionSummary {
  session_id: string;
  total_evaluations: number;
  average_scores: {
    overall: number;
    groundedness: number;
    relevance: number;
    coherence: number;
    fluency: number;
  };
  evaluator_types_used: string[];
  rag_methods_used: string[];
  evaluation_duration_range: {
    min: number;
    max: number;
    avg: number;
  };
  top_questions: Array<{
    question_id: string;
    question: string;
    score: number;
  }>;
}

export const EvaluationAnalytics: React.FC = () => {
  const [analytics, setAnalytics] = useState<EvaluationAnalytics | null>(null);
  const [selectedDays, setSelectedDays] = useState<number>(7);
  const [selectedEvaluatorType, setSelectedEvaluatorType] = useState<string>('all');
  const [selectedRagMethod, setSelectedRagMethod] = useState<string>('all');
  const [selectedSession, setSelectedSession] = useState<string>('');
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        days: selectedDays,
        evaluator_type: selectedEvaluatorType,
        rag_method: selectedRagMethod
      };

      const data = await apiService.getEvaluationAnalytics(params) as EvaluationAnalytics;
      setAnalytics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analytics');
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionSummary = async (sessionId: string) => {
    if (!sessionId) return;
    
    try {
      const data = await apiService.getSessionEvaluationSummary(sessionId) as SessionSummary;
      setSessionSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch session summary');
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, [selectedDays, selectedEvaluatorType, selectedRagMethod]);

  useEffect(() => {
    if (selectedSession) {
      fetchSessionSummary(selectedSession);
    }
  }, [selectedSession]);

  const formatScore = (score: number | null | undefined): string => {
    if (score === null || score === undefined || isNaN(score)) return 'N/A';
    return `${(score * 100).toFixed(1)}%`;
  };

  const getScoreColor = (score: number | null | undefined): string => {
    if (score === null || score === undefined || isNaN(score)) return 'text-gray-500';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

  if (loading && !analytics) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-2 border-primary border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Evaluation Analytics</h2>
          <p className="text-muted-foreground">Monitor and analyze QA evaluation performance</p>
        </div>
        <Button onClick={fetchAnalytics} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh Data'}
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium">Time Period</label>
              <Select value={selectedDays.toString()} onValueChange={(value) => setSelectedDays(parseInt(value))}>
                <SelectTrigger>
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
            
            <div>
              <label className="text-sm font-medium">Evaluator Type</label>
              <Select value={selectedEvaluatorType} onValueChange={setSelectedEvaluatorType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="foundry">Azure AI Foundry</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-sm font-medium">RAG Method</label>
              <Select value={selectedRagMethod} onValueChange={setSelectedRagMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Methods</SelectItem>
                  <SelectItem value="agent">Agent</SelectItem>
                  <SelectItem value="traditional">Traditional</SelectItem>
                  <SelectItem value="agentic-vector">Agentic Vector</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-sm font-medium">Session Analysis</label>
              <input
                type="text"
                placeholder="Enter session ID"
                value={selectedSession}
                onChange={(e) => setSelectedSession(e.target.value)}
                className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {analytics && (
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="sessions">Sessions</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center">
                    <FileText className="h-8 w-8 text-blue-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Total Evaluations</p>
                      <p className="text-2xl font-bold">{analytics.total_evaluations}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center">
                    <Target className="h-8 w-8 text-green-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Avg Overall Score</p>
                      <p className={`text-2xl font-bold ${getScoreColor(analytics.average_scores.overall)}`}>
                        {formatScore(analytics.average_scores.overall)}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center">
                    <Activity className="h-8 w-8 text-purple-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Foundry vs Custom</p>
                      <p className="text-2xl font-bold">
                        {analytics.evaluator_distribution.foundry}:{analytics.evaluator_distribution.custom}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center">
                    <Users className="h-8 w-8 text-orange-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Top Sessions</p>
                      <p className="text-2xl font-bold">{analytics.top_performing_sessions.length}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Score Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Average Scores by Metric</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Groundedness</span>
                        <span className={getScoreColor(analytics.average_scores.groundedness)}>
                          {formatScore(analytics.average_scores.groundedness)}
                        </span>
                      </div>
                      <Progress value={(analytics.average_scores.groundedness || 0) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Relevance</span>
                        <span className={getScoreColor(analytics.average_scores.relevance)}>
                          {formatScore(analytics.average_scores.relevance)}
                        </span>
                      </div>
                      <Progress value={(analytics.average_scores.relevance || 0) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Coherence</span>
                        <span className={getScoreColor(analytics.average_scores.coherence)}>
                          {formatScore(analytics.average_scores.coherence)}
                        </span>
                      </div>
                      <Progress value={(analytics.average_scores.coherence || 0) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Fluency</span>
                        <span className={getScoreColor(analytics.average_scores.fluency)}>
                          {formatScore(analytics.average_scores.fluency)}
                        </span>
                      </div>
                      <Progress value={(analytics.average_scores.fluency || 0) * 100} className="h-2" />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Score Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={analytics.score_distribution}
                        dataKey="count"
                        nameKey="range"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                      >
                        {analytics.score_distribution.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>RAG Method Performance Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.rag_method_performance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="method" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'avg_score' ? formatScore(value as number) : value,
                        name === 'avg_score' ? 'Average Score' : 'Count'
                      ]}
                    />
                    <Bar dataKey="avg_score" fill="#8884d8" name="Average Score" />
                    <Bar dataKey="count" fill="#82ca9d" name="Evaluation Count" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Performing Sessions</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {analytics.top_performing_sessions.map((session, index) => (
                      <div key={session.session_id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <p className="font-medium">Session #{index + 1}</p>
                          <p className="text-sm text-muted-foreground">ID: {session.session_id.substring(0, 8)}...</p>
                          <p className="text-sm text-muted-foreground">{session.evaluation_count} evaluations</p>
                        </div>
                        <Badge variant="outline" className={getScoreColor(session.avg_score)}>
                          {formatScore(session.avg_score)}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Trends Tab */}
          <TabsContent value="trends" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Daily Evaluation Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={analytics.daily_trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'avg_score' ? formatScore(value as number) : value,
                        name === 'avg_score' ? 'Average Score' : 'Evaluations'
                      ]}
                    />
                    <Area type="monotone" dataKey="evaluations" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                    <Area type="monotone" dataKey="avg_score" stackId="2" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Sessions Tab */}
          <TabsContent value="sessions" className="space-y-4">
            {sessionSummary && (
              <Card>
                <CardHeader>
                  <CardTitle>Session Analysis: {selectedSession.substring(0, 16)}...</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Total Evaluations</p>
                      <p className="text-2xl font-bold">{sessionSummary.total_evaluations}</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Average Overall Score</p>
                      <p className={`text-2xl font-bold ${getScoreColor(sessionSummary.average_scores.overall)}`}>
                        {formatScore(sessionSummary.average_scores.overall)}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Evaluator Types Used</p>
                      <div className="flex gap-1">
                        {sessionSummary.evaluator_types_used.map(type => (
                          <Badge key={type} variant="outline">{type}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h4 className="font-medium">Top Questions in Session</h4>
                    <ScrollArea className="h-48">
                      <div className="space-y-2">
                        {sessionSummary.top_questions.map((q, index) => (
                          <div key={q.question_id} className="p-3 border rounded-lg">
                            <div className="flex justify-between items-start">
                              <div className="flex-1">
                                <p className="text-sm font-medium">Question #{index + 1}</p>
                                <p className="text-sm text-muted-foreground">{q.question}</p>
                              </div>
                              <Badge variant="outline" className={getScoreColor(q.score)}>
                                {formatScore(q.score)}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
