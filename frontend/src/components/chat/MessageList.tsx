import React from 'react';
import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { User, Bot, ExternalLink, FileText } from 'lucide-react';
import { Message, Citation } from './ChatContainer';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  onCitationClick: (citations: Citation[]) => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  onCitationClick,
}) => {
  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderCitations = (citations: Citation[]) => {
    if (!citations || citations.length === 0) return null;

    return (
      <div className="mt-3 space-y-2">
        <div className="text-sm font-medium text-muted-foreground">Sources:</div>
        <div className="flex flex-wrap gap-2">
          {citations.map((citation) => (
            <Button
              key={citation.id}
              variant="outline"
              size="sm"
              className="h-auto p-2 text-xs"
              onClick={() => onCitationClick([citation])}
            >
              <FileText className="w-3 h-3 mr-1" />
              <div className="text-left">
                <div className="font-medium">{citation.documentTitle}</div>
                {citation.pageNumber && (
                  <div className="text-muted-foreground">Page {citation.pageNumber}</div>
                )}
              </div>
              <Badge
                variant={citation.confidence === 'high' ? 'default' : 'secondary'}
                className="ml-2"
              >
                {citation.confidence}
              </Badge>
            </Button>
          ))}
        </div>
        {citations.length > 1 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCitationClick(citations)}
            className="text-xs"
          >
            <ExternalLink className="w-3 h-3 mr-1" />
            View all {citations.length} sources
          </Button>
        )}
      </div>
    );
  };

  const renderMessageContent = (content: string, citations?: Citation[]) => {
    if (!citations || citations.length === 0) {
      return <div className="prose prose-sm max-w-none">{content}</div>;
    }

    let processedContent = content;
    const citationMap = new Map<string, Citation>();
    
    citations.forEach((citation, index) => {
      citationMap.set(citation.id, citation);
      const citationNumber = index + 1;
      const citationRegex = new RegExp(`\\[${citation.id}\\]`, 'g');
      processedContent = processedContent.replace(
        citationRegex,
        `<sup class="citation-link cursor-pointer text-blue-600 hover:text-blue-800" data-citation-id="${citation.id}">[${citationNumber}]</sup>`
      );
    });

    return (
      <div 
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: processedContent }}
        onClick={(e) => {
          const target = e.target as HTMLElement;
          if (target.classList.contains('citation-link')) {
            const citationId = target.getAttribute('data-citation-id');
            if (citationId && citationMap.has(citationId)) {
              onCitationClick([citationMap.get(citationId)!]);
            }
          }
        }}
      />
    );
  };

  return (
    <div className="space-y-4">
      {messages.map((message) => (
        <Card key={message.id} className="p-4">
          <div className="flex gap-3">
            <Avatar className="w-8 h-8">
              <AvatarFallback>
                {message.role === 'user' ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </AvatarFallback>
            </Avatar>
            
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2">
                <span className="font-medium">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </span>
                <span className="text-xs text-muted-foreground">
                  {formatTimestamp(message.timestamp)}
                </span>
                {message.metadata?.model && (
                  <Badge variant="secondary" className="text-xs">
                    {message.metadata.model}
                  </Badge>
                )}
              </div>
              
              {renderMessageContent(message.content, message.citations)}
              
              {message.citations && renderCitations(message.citations)}
              
              {message.metadata && (
                <div className="flex gap-4 text-xs text-muted-foreground">
                  {message.metadata.tokens && (
                    <span>Tokens: {message.metadata.tokens}</span>
                  )}
                  {message.metadata.responseTime && (
                    <span>Response time: {message.metadata.responseTime}ms</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </Card>
      ))}
      
      {isLoading && (
        <Card className="p-4">
          <div className="flex gap-3">
            <Avatar className="w-8 h-8">
              <AvatarFallback>
                <Bot className="w-4 h-4" />
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2">
                <span className="font-medium">Assistant</span>
                <span className="text-xs text-muted-foreground">thinking...</span>
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
