import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { X, HelpCircle, ArrowRight } from 'lucide-react';

interface QuestionDecompositionProps {
  decomposition: {
    originalQuestion: string;
    subQuestions: string[];
    reasoning: string;
  };
  onClose: () => void;
}

export const QuestionDecomposition: React.FC<QuestionDecompositionProps> = ({
  decomposition,
  onClose,
}) => {
  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center space-x-2">
            <HelpCircle className="h-5 w-5" />
            <span>Question Decomposition</span>
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          Breaking down complex questions into researchable components
        </p>
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden space-y-4">
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-foreground">Original Question</h4>
          <div className="bg-muted/30 p-3 rounded-lg text-sm text-foreground text-left">
            {decomposition.originalQuestion}
          </div>
        </div>

        <Separator />

        <div className="space-y-3">
          <h4 className="text-sm font-medium text-foreground flex items-center space-x-2">
            <ArrowRight className="h-4 w-4" />
            <span>Sub-Questions ({decomposition.subQuestions.length})</span>
          </h4>
          
          <ScrollArea className="h-48">
            <div className="space-y-2">
              {decomposition.subQuestions.map((subQuestion, index) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 p-3 bg-background border rounded-lg"
                >
                  <div className="flex-shrink-0 w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-medium">
                    {index + 1}
                  </div>
                  <div className="flex-1 text-sm text-foreground text-left">
                    {subQuestion}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>

        <Separator />

        <div className="space-y-2">
          <h4 className="text-sm font-medium text-foreground">Decomposition Reasoning</h4>
          <ScrollArea className="h-32">
            <div className="bg-muted/20 p-3 rounded-lg text-sm text-muted-foreground leading-relaxed">
              {decomposition.reasoning}
            </div>
          </ScrollArea>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-start space-x-2">
            <HelpCircle className="h-4 w-4 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-blue-800">
              <p className="font-medium mb-1">How Question Decomposition Helps:</p>
              <ul className="list-disc list-inside space-y-0.5 text-blue-700">
                <li>Breaks complex queries into focused research areas</li>
                <li>Enables more targeted document retrieval</li>
                <li>Improves answer accuracy and completeness</li>
                <li>Provides transparency in the analysis process</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
