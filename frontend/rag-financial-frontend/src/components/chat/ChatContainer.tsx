import React, { useState, useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { CitationPanel } from './CitationPanel';
import { SessionHistory } from './SessionHistory';
import { ModelSettings } from '../shared/ModelConfiguration';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
  metadata?: {
    model?: string;
    tokens?: number;
    responseTime?: number;
  };
}

export interface Citation {
  id: string;
  content: string;
  source: string;
  documentId: string;
  documentTitle: string;
  pageNumber?: number;
  sectionTitle?: string;
  confidence: 'high' | 'medium' | 'low';
  url?: string;
}



export interface SessionInfo {
  session_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  user_id?: string;
}

interface ChatContainerProps {
  modelSettings: ModelSettings;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({ modelSettings }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([]);
  const [showCitationPanel, setShowCitationPanel] = useState(false);
  const [showSessionHistory, setShowSessionHistory] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const sessionId = localStorage.getItem('currentSessionId') || generateSessionId();
    setCurrentSessionId(sessionId);
    localStorage.setItem('currentSessionId', sessionId);
    
    loadSessionHistory();
    
    if (sessionId) {
      loadSessionMessages(sessionId);
    }
  }, []);

  const generateSessionId = (): string => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const loadSessionHistory = async () => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/chat/sessions`);
      if (response.ok) {
        const sessionData = await response.json();
        setSessions(sessionData);
      }
    } catch (error) {
      console.error('Error loading session history:', error);
    }
  };

  const loadSessionMessages = async (sessionId: string) => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/chat/sessions/${sessionId}/messages`);
      if (response.ok) {
        const messagesData = await response.json();
        const formattedMessages: Message[] = messagesData.map((msg: any) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
          citations: msg.citations || [],
          metadata: msg.metadata || {}
        }));
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Error loading session messages:', error);
    }
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': currentSessionId,
        },
        body: JSON.stringify({
          message: content,
          chat_model: modelSettings.selectedModel,
          embedding_model: modelSettings.embeddingModel,
          temperature: modelSettings.temperature,
          max_tokens: modelSettings.maxTokens,
          search_type: modelSettings.searchType,
          exercise_type: 'context_aware_generation',
          session_id: currentSessionId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        citations: data.citations || [],
        metadata: {
          model: data.model,
          tokens: data.tokens,
          responseTime: data.responseTime,
        },
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCitationClick = (citations: Citation[]) => {
    setSelectedCitations(citations);
    setShowCitationPanel(true);
  };

  const handleSessionSelect = async (sessionId: string) => {
    setCurrentSessionId(sessionId);
    localStorage.setItem('currentSessionId', sessionId);
    await loadSessionMessages(sessionId);
    setShowSessionHistory(false);
  };

  const handleNewSession = () => {
    const newSessionId = generateSessionId();
    setCurrentSessionId(newSessionId);
    localStorage.setItem('currentSessionId', newSessionId);
    setMessages([]);
    setShowSessionHistory(false);
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/chat/sessions/${sessionId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        await loadSessionHistory();
        if (sessionId === currentSessionId) {
          handleNewSession();
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      <ResizablePanelGroup direction="horizontal" className="min-h-screen">
        <ResizablePanel defaultSize={75} minSize={50}>
          <div className="flex flex-col h-full">
            
            <ScrollArea className="flex-1 p-4">
              <MessageList
                messages={messages}
                isLoading={isLoading}
                onCitationClick={handleCitationClick}
              />
              <div ref={messagesEndRef} />
            </ScrollArea>
            
            <div className="border-t p-4">
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={() => setShowSessionHistory(!showSessionHistory)}
                  className="px-3 py-1 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80"
                >
                  {showSessionHistory ? 'Hide History' : 'Show History'}
                </button>
                <button
                  onClick={handleNewSession}
                  className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  New Session
                </button>
                <span className="text-xs text-muted-foreground">
                  Session: {currentSessionId.split('_')[1]}
                </span>
              </div>
              <MessageInput
                onSendMessage={handleSendMessage}
                disabled={isLoading}
                placeholder="Ask about financial documents..."
              />
            </div>
          </div>
        </ResizablePanel>
        
        {(showCitationPanel || showSessionHistory) && (
          <>
            <ResizableHandle />
            <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
              {showCitationPanel && (
                <CitationPanel
                  citations={selectedCitations}
                  onClose={() => setShowCitationPanel(false)}
                />
              )}
              {showSessionHistory && (
                <SessionHistory
                  sessions={sessions}
                  currentSessionId={currentSessionId}
                  onSessionSelect={handleSessionSelect}
                  onNewSession={handleNewSession}
                  onDeleteSession={handleDeleteSession}
                  onClose={() => setShowSessionHistory(false)}
                />
              )}
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </div>
  );
};
