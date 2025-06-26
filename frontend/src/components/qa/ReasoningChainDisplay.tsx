import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Progress } from '@/components/ui/progress';
import { ChevronDown, ChevronRight, Search, Brain, FileText, Shield, Clock } from 'lucide-react';

interface ReasoningStep {
  step_number: number;
  description: string;
  action_type: string;
  sources_consulted: string[];
  confidence: number;
  duration_ms: number;
  output: string;
  metadata: { [key: string]: any };
}

interface ReasoningChain {
  question_id: string;
  question: string;
  reasoning_steps: ReasoningStep[];
  total_duration_ms: number;
  final_confidence: number;
  session_id: string;
  timestamp: string;
}

interface ReasoningChainDisplayProps {
  reasoningChain: ReasoningChain;
  isVisible: boolean;
  onClose: () => void;
}

export const ReasoningChainDisplay: React.FC<ReasoningChainDisplayProps> = ({
  reasoningChain,
  isVisible,
  onClose
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([1])); // Expand first step by default

  if (!isVisible) return null;
  
  // Add debugging and error handling
  if (!reasoningChain) {
    console.error('ReasoningChainDisplay: reasoningChain is null or undefined');
    return <div>Error: No reasoning chain data available</div>;
  }
  
  if (!reasoningChain.reasoning_steps) {
    console.error('ReasoningChainDisplay: reasoning_steps is undefined', reasoningChain);
    return <div>Error: No reasoning steps available</div>;
  }

  const getActionIcon = (actionType: string) => {
    switch (actionType.toLowerCase()) {
      case 'search':
        return <Search className="h-4 w-4" />;
      case 'analyze':
        return <Brain className="h-4 w-4" />;
      case 'synthesize':
        return <FileText className="h-4 w-4" />;
      case 'verify':
        return <Shield className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getActionColor = (actionType: string): string => {
    switch (actionType.toLowerCase()) {
      case 'search':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'analyze':
        return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'synthesize':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'verify':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const toggleStep = (stepNumber: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepNumber)) {
      newExpanded.delete(stepNumber);
    } else {
      newExpanded.add(stepNumber);
    }
    setExpandedSteps(newExpanded);
  };

  const formatDuration = (durationMs: number): string => {
    if (durationMs < 1000) {
      return `${durationMs}ms`;
    } else if (durationMs < 60000) {
      return `${(durationMs / 1000).toFixed(1)}s`;
    } else {
      const minutes = Math.floor(durationMs / 60000);
      const seconds = Math.floor((durationMs % 60000) / 1000);
      return `${minutes}m ${seconds}s`;
    }
  };

  const calculateStepProgress = (stepIndex: number): number => {
    const totalSteps = reasoningChain.reasoning_steps?.length || 1;
    return ((stepIndex + 1) / totalSteps) * 100;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">AI Reasoning Process</h3>
        <Button variant="outline" size="sm" onClick={onClose}>
          ✕ Close
        </Button>
      </div>

      {/* Overview Card */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Reasoning Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-sm text-muted-foreground bg-muted/30 p-3 rounded">
            <strong>Question:</strong> {reasoningChain.question}
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Total Steps</div>
              <div className="text-lg font-semibold">{reasoningChain.reasoning_steps?.length || 0}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Total Time</div>
              <div className="text-lg font-semibold">{formatDuration(reasoningChain.total_duration_ms)}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Final Confidence</div>
              <div className={`text-lg font-semibold ${getConfidenceColor(reasoningChain.final_confidence)}`}>
                {(reasoningChain.final_confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Reasoning Steps */}
      <div className="space-y-3">
        {(reasoningChain.reasoning_steps || []).map((step, index) => (
          <Card key={step.step_number} className="relative">
            <Collapsible
              open={expandedSteps.has(step.step_number)}
              onOpenChange={() => toggleStep(step.step_number)}
            >
              <CollapsibleTrigger asChild>
                <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        {expandedSteps.has(step.step_number) ? (
                          <ChevronDown className="h-4 w-4 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        )}
                        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary font-medium text-sm">
                          {step.step_number}
                        </div>
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <Badge variant="outline" className={getActionColor(step.action_type)}>
                            {getActionIcon(step.action_type)}
                            <span className="ml-1 capitalize">{step.action_type}</span>
                          </Badge>
                          <span className={`text-xs font-medium ${getConfidenceColor(step.confidence)}`}>
                            {(step.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                        <div className="text-sm text-foreground">{step.description}</div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>{formatDuration(step.duration_ms)}</span>
                    </div>
                  </div>
                  
                  {/* Progress indicator */}
                  <div className="mt-2">
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>Step Progress</span>
                      <span>{calculateStepProgress(index).toFixed(0)}% Complete</span>
                    </div>
                    <Progress value={calculateStepProgress(index)} className="h-1" />
                  </div>
                </CardHeader>
              </CollapsibleTrigger>
              
              <CollapsibleContent>
                <CardContent className="pt-0 space-y-3">
                  {/* Step Output */}
                  {step.output && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-muted-foreground">Output:</div>
                      <div className="text-sm bg-muted/30 p-3 rounded text-left">
                        {step.output}
                      </div>
                    </div>
                  )}
                  
                  {/* Sources Consulted */}
                  {step.sources_consulted.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-muted-foreground">
                        Sources Consulted ({step.sources_consulted.length}):
                      </div>
                      <div className="space-y-1">
                        {step.sources_consulted.slice(0, 3).map((source, sourceIndex) => (
                          <div key={sourceIndex} className="text-xs text-muted-foreground bg-muted/20 p-2 rounded">
                            {source.length > 80 ? `${source.substring(0, 80)}...` : source}
                          </div>
                        ))}
                        {step.sources_consulted.length > 3 && (
                          <div className="text-xs text-muted-foreground">
                            +{step.sources_consulted.length - 3} more sources
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Metadata */}
                  {step.metadata && Object.keys(step.metadata).length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs font-medium text-muted-foreground">Additional Details:</div>
                      <div className="text-xs text-muted-foreground bg-muted/20 p-2 rounded">
                        {Object.entries(step.metadata)
                          .filter(([key]) => !['start_time'].includes(key))
                          .map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="capitalize">{key.replace('_', ' ')}:</span>
                              <span>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        ))}
      </div>

      {/* Summary */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20">
        <CardContent className="pt-6">
          <div className="text-sm space-y-2">
            <div className="font-medium text-foreground">Process Summary:</div>
            <div className="text-muted-foreground">
              The AI completed <strong>{reasoningChain.reasoning_steps?.length || 0} reasoning steps</strong> in{' '}
              <strong>{formatDuration(reasoningChain.total_duration_ms)}</strong>, achieving{' '}
              <strong className={getConfidenceColor(reasoningChain.final_confidence)}>
                {(reasoningChain.final_confidence * 100).toFixed(0)}% confidence
              </strong> in the final answer.
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
