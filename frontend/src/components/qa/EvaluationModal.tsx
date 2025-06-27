import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

interface EvaluationResult {
  id: string;
  question_id: string;
  session_id: string;
  evaluator_type: string;
  rag_method: string;
  groundedness_score?: number;
  relevance_score?: number;
  coherence_score?: number;
  fluency_score?: number;
  similarity_score?: number;
  f1_score?: number;
  bleu_score?: number;
  rouge_score?: number;
  overall_score?: number;
  evaluation_model: string;
  evaluation_timestamp: string;
  evaluation_duration_ms?: number;
  question: string;
  answer: string;
  context: string[];
  ground_truth?: string;
  detailed_scores: any;
  reasoning?: string;
  feedback?: string;
  recommendations: string[];
  error?: string;
}

interface EvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  evaluationResult: EvaluationResult | null;
}

export const EvaluationModal: React.FC<EvaluationModalProps> = ({
  isOpen,
  onClose,
  evaluationResult
}) => {
  if (!evaluationResult) return null;

  const getScoreColor = (score: number | undefined) => {
    if (!score) return 'text-muted-foreground';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreLabel = (score: number | undefined) => {
    if (!score) return 'N/A';
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Poor';
  };

  const formatScore = (score: number | undefined) => {
    return score ? (score * 100).toFixed(1) + '%' : 'N/A';
  };

  const scores = [
    { name: 'Overall', value: evaluationResult.overall_score, description: 'Overall evaluation score' },
    { name: 'Groundedness', value: evaluationResult.groundedness_score, description: 'How well the answer is grounded in the provided sources' },
    { name: 'Relevance', value: evaluationResult.relevance_score, description: 'How relevant the answer is to the question' },
    { name: 'Coherence', value: evaluationResult.coherence_score, description: 'How coherent and well-structured the answer is' },
    { name: 'Fluency', value: evaluationResult.fluency_score, description: 'How fluent and natural the answer reads' },
    { name: 'Similarity', value: evaluationResult.similarity_score, description: 'Semantic similarity to expected answer' },
    { name: 'F1 Score', value: evaluationResult.f1_score, description: 'F1 score for answer precision and recall' },
    { name: 'BLEU Score', value: evaluationResult.bleu_score, description: 'BLEU score for translation-like quality' },
    { name: 'ROUGE Score', value: evaluationResult.rouge_score, description: 'ROUGE score for summarization quality' },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <span>Evaluation Results</span>
            <Badge variant="outline" className="text-xs">
              {evaluationResult.evaluator_type === 'foundry' ? 'Azure AI Foundry' : 'Custom Evaluator'}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {evaluationResult.evaluation_model}
            </Badge>
          </DialogTitle>
        </DialogHeader>
        
        <ScrollArea className="max-h-[70vh]">
          <div className="space-y-6 pr-4">
            {/* Error Display */}
            {evaluationResult.error && (
              <Card className="border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20">
                <CardHeader>
                  <CardTitle className="text-red-800 dark:text-red-300 text-sm">
                    Evaluation Error
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-red-700 dark:text-red-400">
                    {evaluationResult.error}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Overall Score */}
            {evaluationResult.overall_score && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Overall Evaluation Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center space-x-4">
                    <div className="text-3xl font-bold" style={{ color: getScoreColor(evaluationResult.overall_score) }}>
                      {formatScore(evaluationResult.overall_score)}
                    </div>
                    <div className="flex-1">
                      <Progress 
                        value={(evaluationResult.overall_score || 0) * 100} 
                        className="h-3"
                      />
                      <p className="text-sm text-muted-foreground mt-1">
                        {getScoreLabel(evaluationResult.overall_score)}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Detailed Scores */}
            <Card>
              <CardHeader>
                <CardTitle>Detailed Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {scores.filter(score => score.value !== undefined).map((score) => (
                    <div key={score.name} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">{score.name}</span>
                        <span className={`text-sm font-semibold ${getScoreColor(score.value)}`}>
                          {formatScore(score.value)}
                        </span>
                      </div>
                      <Progress value={(score.value || 0) * 100} className="h-2" />
                      <p className="text-xs text-muted-foreground">
                        {score.description}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Reasoning */}
            {evaluationResult.reasoning && (
              <Card>
                <CardHeader>
                  <CardTitle>Evaluation Reasoning</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-foreground whitespace-pre-wrap">
                    {evaluationResult.reasoning}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Feedback */}
            {evaluationResult.feedback && (
              <Card>
                <CardHeader>
                  <CardTitle>Detailed Feedback</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-foreground whitespace-pre-wrap">
                    {evaluationResult.feedback}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            {evaluationResult.recommendations && evaluationResult.recommendations.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-2">
                    {evaluationResult.recommendations.map((rec, index) => (
                      <li key={index} className="text-sm text-foreground">
                        {rec}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* Evaluation Metadata */}
            <Card>
              <CardHeader>
                <CardTitle>Evaluation Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Evaluator Type:</span>
                    <span className="ml-2">{evaluationResult.evaluator_type}</span>
                  </div>
                  <div>
                    <span className="font-medium">RAG Method:</span>
                    <span className="ml-2">{evaluationResult.rag_method}</span>
                  </div>
                  <div>
                    <span className="font-medium">Model Used:</span>
                    <span className="ml-2">{evaluationResult.evaluation_model}</span>
                  </div>
                  <div>
                    <span className="font-medium">Duration:</span>
                    <span className="ml-2">
                      {evaluationResult.evaluation_duration_ms ? 
                        `${evaluationResult.evaluation_duration_ms}ms` : 'N/A'}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium">Timestamp:</span>
                    <span className="ml-2">
                      {new Date(evaluationResult.evaluation_timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Question and Answer Context */}
            <Card>
              <CardHeader>
                <CardTitle>Evaluation Context</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium text-sm mb-2">Question:</h4>
                  <p className="text-sm bg-muted p-3 rounded">
                    {evaluationResult.question}
                  </p>
                </div>
                <Separator />
                <div>
                  <h4 className="font-medium text-sm mb-2">Answer:</h4>
                  <div className="text-sm bg-muted p-3 rounded max-h-40 overflow-y-auto">
                    {evaluationResult.answer}
                  </div>
                </div>
                {evaluationResult.context && evaluationResult.context.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <h4 className="font-medium text-sm mb-2">Context Sources ({evaluationResult.context.length}):</h4>
                      <div className="space-y-2 max-h-40 overflow-y-auto">
                        {evaluationResult.context.map((ctx, index) => (
                          <div key={index} className="text-xs bg-muted/50 p-2 rounded">
                            {ctx.length > 200 ? ctx.substring(0, 200) + '...' : ctx}
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};
