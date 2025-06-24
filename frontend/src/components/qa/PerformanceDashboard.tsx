import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Clock, TrendingUp, Target, CheckCircle } from 'lucide-react';

interface PerformanceBenchmark {
  question_id: string;
  question: string;
  complexity_score: number;
  estimated_manual_time: number;
  ai_processing_time: number;
  efficiency_gain: number;
  source_count: number;
  accuracy_score: number;
  confidence_score: number;
  verification_level: string;
  session_id: string;
  timestamp: string;
}

interface PerformanceMetrics {
  total_questions: number;
  average_efficiency_gain: number;
  average_accuracy_score: number;
  average_processing_time: number;
  complexity_breakdown: { [key: number]: number };
  time_saved_minutes: number;
  session_id: string;
  timestamp: string;
}

interface PerformanceDashboardProps {
  benchmark?: PerformanceBenchmark;
  sessionMetrics?: PerformanceMetrics;
  isVisible: boolean;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  benchmark,
  sessionMetrics,
  isVisible
}) => {
  if (!isVisible) return null;

  const getComplexityLabel = (score: number): string => {
    const labels = {
      1: 'Simple',
      2: 'Medium',
      3: 'Complex',
      4: 'Very Complex',
      5: 'Extremely Complex'
    };
    return labels[score as keyof typeof labels] || 'Unknown';
  };

  const getComplexityColor = (score: number): string => {
    if (score <= 2) return 'bg-green-100 text-green-800 border-green-200';
    if (score <= 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const getEfficiencyColor = (gain: number): string => {
    if (gain >= 80) return 'text-green-600';
    if (gain >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatTime = (minutes: number): string => {
    if (minutes < 1) {
      return `${Math.round(minutes * 60)}s`;
    } else if (minutes < 60) {
      return `${minutes.toFixed(1)}m`;
    } else {
      const hours = Math.floor(minutes / 60);
      const remainingMinutes = Math.round(minutes % 60);
      return `${hours}h ${remainingMinutes}m`;
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-foreground">Performance Analytics</h3>
      
      {/* Current Question Benchmark */}
      {benchmark && (
        <Card className="border-l-4 border-l-blue-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center space-x-2">
              <Target className="h-4 w-4" />
              <span>Current Question Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Complexity</span>
                <Badge variant="outline" className={getComplexityColor(benchmark.complexity_score)}>
                  {getComplexityLabel(benchmark.complexity_score)} ({benchmark.complexity_score}/5)
                </Badge>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Efficiency Gain</span>
                <span className={`text-sm font-semibold ${getEfficiencyColor(benchmark.efficiency_gain)}`}>
                  {benchmark.efficiency_gain.toFixed(1)}%
                </span>
              </div>
              
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>AI Processing</span>
                  <span>Manual Estimate</span>
                </div>
                <div className="flex justify-between text-sm font-medium">
                  <span className="text-green-600">{formatTime(benchmark.ai_processing_time)}</span>
                  <span className="text-muted-foreground">{formatTime(benchmark.estimated_manual_time)}</span>
                </div>
                <Progress 
                  value={(benchmark.ai_processing_time / benchmark.estimated_manual_time) * 100} 
                  className="h-2"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">Sources Analyzed</div>
                  <div className="text-sm font-medium">{benchmark.source_count}</div>
                </div>
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">Accuracy Score</div>
                  <div className="text-sm font-medium">{(benchmark.accuracy_score * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Session Performance Summary */}
      {sessionMetrics && sessionMetrics.total_questions > 0 && (
        <Card className="border-l-4 border-l-green-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center space-x-2">
              <TrendingUp className="h-4 w-4" />
              <span>Session Summary</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-1">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span className="text-xs text-muted-foreground">Questions Answered</span>
                </div>
                <div className="text-lg font-semibold text-foreground">
                  {sessionMetrics.total_questions}
                </div>
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center space-x-1">
                  <Clock className="h-3 w-3 text-blue-600" />
                  <span className="text-xs text-muted-foreground">Time Saved</span>
                </div>
                <div className="text-lg font-semibold text-green-600">
                  {formatTime(sessionMetrics.time_saved_minutes)}
                </div>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Average Efficiency Gain</span>
                <span className={`text-sm font-semibold ${getEfficiencyColor(sessionMetrics.average_efficiency_gain)}`}>
                  {sessionMetrics.average_efficiency_gain.toFixed(1)}%
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Average Accuracy</span>
                <span className="text-sm font-semibold">
                  {(sessionMetrics.average_accuracy_score * 100).toFixed(0)}%
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Avg Processing Time</span>
                <span className="text-sm font-semibold">
                  {formatTime(sessionMetrics.average_processing_time)}
                </span>
              </div>
            </div>
            
            {/* Complexity Breakdown */}
            {Object.keys(sessionMetrics.complexity_breakdown).length > 0 && (
              <div className="space-y-2 pt-2 border-t">
                <div className="text-xs text-muted-foreground">Question Complexity Distribution</div>
                <div className="space-y-1">
                  {Object.entries(sessionMetrics.complexity_breakdown).map(([complexity, count]) => (
                    <div key={complexity} className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">
                        {getComplexityLabel(parseInt(complexity))}
                      </span>                      <Badge variant="outline" className="text-xs">
                        {count}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
      
      {/* Performance Insights */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Key Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            {benchmark && (
              <>
                <div className="flex items-start space-x-2">
                  <div className="h-2 w-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    Research completed <strong>{benchmark.efficiency_gain.toFixed(0)}% faster</strong> than 
                    traditional manual research methods
                  </div>
                </div>
                
                <div className="flex items-start space-x-2">
                  <div className="h-2 w-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    Analyzed <strong>{benchmark.source_count} sources</strong> with 
                    <strong> {(benchmark.confidence_score * 100).toFixed(0)}% confidence</strong> in results
                  </div>
                </div>
              </>
            )}
            
            {sessionMetrics && sessionMetrics.total_questions > 1 && (
              <div className="flex items-start space-x-2">
                <div className="h-2 w-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  Session efficiency: <strong>{formatTime(sessionMetrics.time_saved_minutes)} saved</strong> across {sessionMetrics.total_questions} questions
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
