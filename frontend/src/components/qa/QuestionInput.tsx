import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';

interface QuestionInputProps {
  onAskQuestion: (question: string, verificationLevel: 'basic' | 'thorough' | 'comprehensive') => void;
  disabled?: boolean;
}

export const QuestionInput: React.FC<QuestionInputProps> = ({ onAskQuestion, disabled }) => {
  const [question, setQuestion] = useState('');
  const [verificationLevel, setVerificationLevel] = useState<'basic' | 'thorough' | 'comprehensive'>('thorough');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !disabled) {
      onAskQuestion(question.trim(), verificationLevel);
      setQuestion('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleExampleClick = (exampleQuestion: string) => {
    setQuestion(exampleQuestion);
  };

  const exampleQuestions = [
    "What are the key financial risks mentioned in Apple's latest 10-K filing?",
    "Compare and Contrast the Risk factors between Apple and Microsoft for 2023",
    "How has Microsoft's revenue growth compared to industry benchmarks over the past 3 years?",
    "What regulatory compliance issues are affecting the banking sector according to recent reports?"
  ];

  const getVerificationDescription = (level: string) => {
    switch (level) {
      case 'basic':
        return 'Quick analysis with basic source checking';
      case 'thorough':
        return 'Comprehensive analysis with detailed source verification';
      case 'comprehensive':
        return 'Deep analysis with extensive credibility assessment';
      default:
        return '';
    }
  };

  const getVerificationColor = (level: string) => {
    switch (level) {
      case 'basic':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300 dark:border-yellow-800';
      case 'thorough':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800';
      case 'comprehensive':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-700';
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="space-y-2">
        <label htmlFor="verification-level" className="text-sm font-medium text-foreground">
          Verification Level
        </label>
        <Select
          value={verificationLevel}
          onValueChange={(value: 'basic' | 'thorough' | 'comprehensive') => setVerificationLevel(value)}
          disabled={disabled}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select verification level" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="basic">
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className={getVerificationColor('basic')}>
                  Basic
                </Badge>
                <span className="text-sm text-muted-foreground">Fast analysis</span>
              </div>
            </SelectItem>
            <SelectItem value="thorough">
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className={getVerificationColor('thorough')}>
                  Thorough
                </Badge>
                <span className="text-sm text-muted-foreground">Balanced approach</span>
              </div>
            </SelectItem>
            <SelectItem value="comprehensive">
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className={getVerificationColor('comprehensive')}>
                  Comprehensive
                </Badge>
                <span className="text-sm text-muted-foreground">Deep analysis</span>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {getVerificationDescription(verificationLevel)}
        </p>
      </div>

      <div className="space-y-2">
        <label htmlFor="question" className="text-sm font-medium text-foreground">
          Financial Question
        </label>
        <Textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a complex financial question that requires analysis and source verification..."
          className="min-h-[100px] resize-none"
          disabled={disabled}
        />
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Press Enter to submit, Shift+Enter for new line
          </p>
          <span className="text-xs text-muted-foreground">
            {question.length}/2000
          </span>
        </div>
      </div>

      <Button
        type="submit"
        disabled={!question.trim() || disabled}
        className="w-full"
      >
        {disabled ? 'Processing...' : 'Ask Question'}
      </Button>

      <div className="text-xs text-muted-foreground space-y-2">
        <p><strong>Example questions:</strong></p>
        <div className="space-y-1 ml-2">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              type="button"
              onClick={() => handleExampleClick(example)}
              disabled={disabled}
              className="block text-left text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 hover:underline cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              • {example}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-2 italic">
          Click any example above to populate the question field
        </p>
      </div>
    </form>
  );
};
