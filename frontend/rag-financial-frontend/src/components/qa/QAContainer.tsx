import React, { useState, useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { QuestionInput } from './QuestionInput';
import { AnswerDisplay } from './AnswerDisplay';
import { SourceVerification } from './SourceVerification';
import { QuestionDecomposition } from './QuestionDecomposition';
import { AgentServiceStatus } from './AgentServiceStatus';
import { ModelSettings } from '../shared/ModelConfiguration';
import { apiService } from '../../services/api';

export interface QAQuestion {
  id: string;
  question: string;
  timestamp: Date;
  verificationLevel: 'basic' | 'thorough' | 'comprehensive';
}

export interface QAAnswer {
  id: string;
  questionId: string;
  answer: string;
  confidenceScore: number;
  timestamp: Date;
  citations: QACitation[];
  subQuestions: string[];
  verificationDetails: {
    overallCredibilityScore: number;
    verifiedSourcesCount: number;
    totalSourcesCount: number;
    verificationSummary: string;
  };
  metadata: {
    model?: string;
    tokens?: number;
    responseTime?: number;
    verificationLevel?: string;
    agentServiceUsed?: boolean;
    agentId?: string;
    threadId?: string;
  };
}

export interface QACitation {
  id: string;
  content: string;
  source: string;
  documentId: string;
  documentTitle: string;
  pageNumber?: number;
  sectionTitle?: string;
  confidence: 'high' | 'medium' | 'low';
  url?: string;
  credibilityScore: number;
}

export interface VerifiedSource {
  sourceId: string;
  url: string;
  title: string;
  content: string;
  credibilityScore: number;
  credibilityExplanation: string;
  trustIndicators: string[];
  redFlags: string[];
  verificationStatus: 'verified' | 'questionable' | 'unverified';
}

interface QAContainerProps {
  modelSettings: ModelSettings;
}

export const QAContainer: React.FC<QAContainerProps> = ({ modelSettings }) => {
  const [questions, setQuestions] = useState<QAQuestion[]>([]);
  const [answers, setAnswers] = useState<QAAnswer[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [selectedAnswer, setSelectedAnswer] = useState<QAAnswer | null>(null);
  const [showSourceVerification, setShowSourceVerification] = useState(false);
  const [showQuestionDecomposition, setShowQuestionDecomposition] = useState(false);
  const [verifiedSources, setVerifiedSources] = useState<VerifiedSource[]>([]);
  const [decomposedQuestions, setDecomposedQuestions] = useState<{
    originalQuestion: string;
    subQuestions: string[];
    reasoning: string;
  } | null>(null);
  const [showAgentStatus, setShowAgentStatus] = useState(false);
  const [agentServiceConnected, setAgentServiceConnected] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [questions, answers]);

  useEffect(() => {
    const sessionId = localStorage.getItem('qaSessionId') || generateSessionId();
    setCurrentSessionId(sessionId);
    localStorage.setItem('qaSessionId', sessionId);
    
    checkAgentServiceStatus();
  }, []);

  const checkAgentServiceStatus = async () => {
    try {
      await apiService.getQACapabilities();
      setAgentServiceConnected(true);
    } catch (error) {
      console.warn('Agent service check failed:', error);
      setAgentServiceConnected(false);
    }
  };

  const generateSessionId = (): string => {
    return `qa_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const handleAskQuestion = async (
    question: string, 
    verificationLevel: 'basic' | 'thorough' | 'comprehensive'
  ) => {
    const qaQuestion: QAQuestion = {
      id: Date.now().toString(),
      question,
      timestamp: new Date(),
      verificationLevel,
    };

    setQuestions(prev => [...prev, qaQuestion]);
    setIsLoading(true);

    try {
      const data = await apiService.askQuestion({
        question,
        session_id: currentSessionId,
        verification_level: verificationLevel,
      });
      
      const qaAnswer: QAAnswer = {
        id: (Date.now() + 1).toString(),
        questionId: qaQuestion.id,
        answer: data.answer,
        confidenceScore: data.confidence_score || 0.8,
        timestamp: new Date(),
        citations: data.citations || [],
        subQuestions: data.sub_questions || [],
        verificationDetails: {
          overallCredibilityScore: data.verification_details?.overall_credibility_score || 0.5,
          verifiedSourcesCount: data.verification_details?.verified_sources_count || 0,
          totalSourcesCount: data.verification_details?.total_sources_count || 0,
          verificationSummary: data.verification_details?.verification_summary || 'No verification details available',
        },
        metadata: {
          model: data.metadata?.model_used || modelSettings.selectedModel,
          tokens: data.token_usage?.total_tokens,
          responseTime: data.metadata?.response_time,
          verificationLevel: verificationLevel,
          agentServiceUsed: data.metadata?.agent_service_used || agentServiceConnected,
          agentId: data.metadata?.agent_id,
          threadId: data.metadata?.thread_id,
        },
      };

      setAnswers(prev => [...prev, qaAnswer]);
    } catch (error) {
      console.error('Error processing QA request:', error);
      const errorAnswer: QAAnswer = {
        id: (Date.now() + 1).toString(),
        questionId: qaQuestion.id,
        answer: 'Sorry, I encountered an error processing your question. Please try again.',
        confidenceScore: 0,
        timestamp: new Date(),
        citations: [],
        subQuestions: [],
        verificationDetails: {
          overallCredibilityScore: 0,
          verifiedSourcesCount: 0,
          totalSourcesCount: 0,
          verificationSummary: 'Error occurred during processing',
        },
        metadata: {},
      };
      setAnswers(prev => [...prev, errorAnswer]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDecomposeQuestion = async (question: string) => {
    try {
      const data = await apiService.decomposeQuestion({
        question,
        session_id: currentSessionId,
      });
      setDecomposedQuestions({
        originalQuestion: question,
        subQuestions: data.sub_questions,
        reasoning: data.metadata?.reasoning || 'Question decomposed successfully',
      });
      setShowQuestionDecomposition(true);
    } catch (error) {
      console.error('Error decomposing question:', error);
    }
  };

  const handleVerifySources = async (answer: QAAnswer) => {
    try {
      const verifiedSources: VerifiedSource[] = [];
      
      for (const citation of answer.citations) {
        const data = await apiService.verifySource({
          source_url: citation.url || citation.source,
          content: citation.content,
          session_id: currentSessionId,
        });
        
        verifiedSources.push({
          sourceId: citation.id,
          url: citation.url || citation.source,
          title: citation.documentTitle,
          content: citation.content,
          credibilityScore: data.credibility_score,
          credibilityExplanation: data.verification_details?.explanation || 'No explanation provided',
          trustIndicators: data.verification_details?.trust_indicators || [],
          redFlags: data.verification_details?.red_flags || [],
          verificationStatus: data.credibility_score > 0.7 ? 'verified' : 
                            data.credibility_score > 0.4 ? 'questionable' : 'unverified',
        });
      }
      
      setVerifiedSources(verifiedSources);
      setSelectedAnswer(answer);
      setShowSourceVerification(true);
    } catch (error) {
      console.error('Error verifying sources:', error);
    }
  };

  const handleNewSession = () => {
    const newSessionId = generateSessionId();
    setCurrentSessionId(newSessionId);
    localStorage.setItem('qaSessionId', newSessionId);
    setQuestions([]);
    setAnswers([]);
    setSelectedAnswer(null);
    setShowSourceVerification(false);
    setShowQuestionDecomposition(false);
    setShowAgentStatus(false);
    setVerifiedSources([]);
    setDecomposedQuestions(null);
  };

  const handleToggleAgentStatus = () => {
    setShowAgentStatus(!showAgentStatus);
  };

  const handleRefreshAgentStatus = () => {
    checkAgentServiceStatus();
  };

  return (
    <div className="flex h-screen bg-background">
      <ResizablePanelGroup direction="horizontal" className="min-h-screen">
        <ResizablePanel defaultSize={75} minSize={50}>
          <div className="flex flex-col h-full">
            <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
              <div className="space-y-6">
                {questions.map((question, index) => {
                  const answer = answers.find(a => a.questionId === question.id);
                  return (
                    <div key={question.id} className="space-y-4">
                      <div className="bg-muted/50 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-muted-foreground">
                            Question #{index + 1} • {question.verificationLevel} verification
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {question.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-foreground">{question.question}</p>
                        <div className="flex gap-2 mt-2">
                          <button
                            onClick={() => handleDecomposeQuestion(question.question)}
                            className="px-2 py-1 text-xs bg-secondary text-secondary-foreground rounded hover:bg-secondary/80"
                          >
                            Decompose
                          </button>
                        </div>
                      </div>
                      
                      {answer && (
                        <AnswerDisplay
                          answer={answer}
                          onVerifySources={() => handleVerifySources(answer)}
                        />
                      )}
                    </div>
                  );
                })}
                
                {isLoading && (
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      <span className="text-sm text-muted-foreground">
                        Analyzing question and verifying sources...
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
            
            <div className="border-t p-4">
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={handleNewSession}
                  className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  New QA Session
                </button>
                <button
                  onClick={handleToggleAgentStatus}
                  className={`px-3 py-1 text-sm rounded-md hover:opacity-80 ${
                    agentServiceConnected 
                      ? 'bg-green-100 text-green-800 border border-green-200' 
                      : 'bg-red-100 text-red-800 border border-red-200'
                  }`}
                >
                  Agent Status
                </button>
                <span className="text-xs text-muted-foreground">
                  Session: {currentSessionId.split('_')[2]}
                </span>
                {agentServiceConnected && (
                  <span className="text-xs text-green-600 font-medium">
                    ✓ Azure AI Agent Service
                  </span>
                )}
              </div>
              <QuestionInput
                onAskQuestion={handleAskQuestion}
                disabled={isLoading}
              />
            </div>
          </div>
        </ResizablePanel>
        
        {(showSourceVerification || showQuestionDecomposition || showAgentStatus) && (
          <>
            <ResizableHandle />
            <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
              {showSourceVerification && selectedAnswer && (
                <SourceVerification
                  answer={selectedAnswer}
                  verifiedSources={verifiedSources}
                  onClose={() => setShowSourceVerification(false)}
                />
              )}
              {showQuestionDecomposition && decomposedQuestions && (
                <QuestionDecomposition
                  decomposition={decomposedQuestions}
                  onClose={() => setShowQuestionDecomposition(false)}
                />
              )}
              {showAgentStatus && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium">Agent Service Status</h3>
                    <button
                      onClick={() => setShowAgentStatus(false)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <AgentServiceStatus onRefresh={handleRefreshAgentStatus} />
                </div>
              )}
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </div>
  );
};
