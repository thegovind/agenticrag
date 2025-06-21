import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { QAAnswer, QACitation } from './QAContainer';

interface AnswerDisplayProps {
  answer: QAAnswer;
  onVerifySources: () => void;
}

export const AnswerDisplay: React.FC<AnswerDisplayProps> = ({ answer, onVerifySources }) => {
  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300 dark:border-yellow-800';
    return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800';
  };

  const getConfidenceLabel = (score: number) => {
    if (score >= 0.8) return 'High Confidence';
    if (score >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  const getCredibilityColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatCitationContent = (citation: QACitation) => {
    const maxLength = 150;
    return citation.content.length > maxLength 
      ? citation.content.substring(0, maxLength) + '...'
      : citation.content;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Answer</CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className={getConfidenceColor(answer.confidenceScore)}>
              {getConfidenceLabel(answer.confidenceScore)} ({(answer.confidenceScore * 100).toFixed(0)}%)
            </Badge>
            <Badge variant="outline">
              {answer.metadata.verificationLevel || 'thorough'}
            </Badge>
          </div>
        </div>
        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
          <span>{answer.timestamp.toLocaleTimeString()}</span>
          {answer.metadata.model && <span>Model: {answer.metadata.model}</span>}
          {answer.metadata.responseTime && (
            <span>Response: {(answer.metadata.responseTime * 1000).toFixed(0)}ms</span>
          )}
          {answer.metadata.tokens && <span>Tokens: {answer.metadata.tokens}</span>}
          {answer.metadata.agentServiceUsed && (
            <span className="text-green-600 font-medium">✓ Agent Service</span>
          )}
        </div>
        {answer.metadata.agentId && (
          <div className="text-xs text-muted-foreground">
            Agent ID: {answer.metadata.agentId}
            {answer.metadata.threadId && ` • Thread: ${answer.metadata.threadId.substring(0, 8)}...`}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="prose prose-sm max-w-none">
          <div className="whitespace-pre-wrap text-foreground leading-relaxed">
            {answer.answer}
          </div>
        </div>

        {answer.subQuestions.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-foreground">Research Sub-Questions</h4>
            <div className="space-y-1">
              {answer.subQuestions.map((subQuestion, index) => (
                <div key={index} className="text-sm text-muted-foreground bg-muted/30 p-2 rounded">
                  {index + 1}. {subQuestion}
                </div>
              ))}
            </div>
          </div>
        )}

        <Separator />

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-foreground">
              Source Verification ({answer.citations.length} sources)
            </h4>
            <Button
              variant="outline"
              size="sm"
              onClick={onVerifySources}
              className="text-xs"
            >
              Verify Sources
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
            <div className="bg-muted/30 p-2 rounded">
              <span className="font-medium">Overall Credibility:</span>
              <span className={`ml-1 font-medium ${getCredibilityColor(answer.verificationDetails.overallCredibilityScore)}`}>
                {(answer.verificationDetails.overallCredibilityScore * 100).toFixed(0)}%
              </span>
            </div>
            <div className="bg-muted/30 p-2 rounded">
              <span className="font-medium">Verified Sources:</span>
              <span className="ml-1">
                {answer.verificationDetails.verifiedSourcesCount}/{answer.verificationDetails.totalSourcesCount}
              </span>
            </div>
          </div>

          {answer.verificationDetails.verificationSummary && (
            <div className="text-xs text-muted-foreground bg-muted/20 p-2 rounded">
              <strong>Verification Summary:</strong> {answer.verificationDetails.verificationSummary}
            </div>
          )}
        </div>

        {answer.citations.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-foreground">Citations</h4>
            <ScrollArea className="h-32">
              <div className="space-y-2">
                {answer.citations.map((citation, index) => (
                  <div key={citation.id} className="text-xs border rounded p-2 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-foreground">
                        [{index + 1}] {citation.documentTitle}
                      </span>
                      <div className="flex items-center space-x-1">
                        <Badge variant="outline" className="text-xs">
                          {citation.confidence}
                        </Badge>
                        {citation.credibilityScore !== undefined && (
                          <span className={`text-xs font-medium ${getCredibilityColor(citation.credibilityScore)}`}>
                            {(citation.credibilityScore * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-muted-foreground">
                      {formatCitationContent(citation)}
                    </div>
                    <div className="flex items-center space-x-2 text-muted-foreground">
                      <span>Source: {citation.source}</span>
                      {citation.pageNumber && <span>Page: {citation.pageNumber}</span>}
                      {citation.sectionTitle && <span>Section: {citation.sectionTitle}</span>}
                    </div>
                    {citation.url && (
                      <a
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline text-xs"
                      >
                        View Source
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
