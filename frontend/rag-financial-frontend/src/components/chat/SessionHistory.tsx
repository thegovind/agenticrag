import React from 'react';
import { ScrollArea } from '../ui/scroll-area';
import { Button } from '../ui/button';
import { X, MessageSquare, Trash2, Plus } from 'lucide-react';

interface SessionInfo {
  session_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  user_id?: string;
}

interface SessionHistoryProps {
  sessions: SessionInfo[];
  currentSessionId: string;
  onSessionSelect: (sessionId: string) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  onClose: () => void;
}

export const SessionHistory: React.FC<SessionHistoryProps> = ({
  sessions,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  onDeleteSession,
  onClose,
}) => {
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return 'Unknown';
    }
  };

  const getSessionDisplayName = (sessionId: string) => {
    const parts = sessionId.split('_');
    if (parts.length >= 2) {
      const timestamp = parseInt(parts[1]);
      if (!isNaN(timestamp)) {
        const date = new Date(timestamp);
        return `Session ${date.toLocaleDateString()}`;
      }
    }
    return `Session ${sessionId.substring(0, 8)}...`;
  };

  return (
    <div className="flex flex-col h-full bg-background border-l">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          <h3 className="font-semibold">Chat History</h3>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="p-4 border-b">
        <Button onClick={onNewSession} className="w-full" size="sm">
          <Plus className="h-4 w-4 mr-2" />
          New Session
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-2">
          {sessions.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No chat sessions yet</p>
              <p className="text-xs">Start a conversation to see your history</p>
            </div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.session_id}
                className={`group p-3 rounded-lg border cursor-pointer transition-colors ${
                  session.session_id === currentSessionId
                    ? 'bg-primary/10 border-primary'
                    : 'hover:bg-muted border-border'
                }`}
                onClick={() => onSessionSelect(session.session_id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <MessageSquare className="h-3 w-3 text-muted-foreground" />
                      <span className="text-sm font-medium truncate">
                        {getSessionDisplayName(session.session_id)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <div>Messages: {session.message_count}</div>
                      <div>Updated: {formatDate(session.updated_at)}</div>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.session_id);
                    }}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      <div className="p-4 border-t">
        <div className="text-xs text-muted-foreground">
          <div className="flex items-center justify-between">
            <span>Total Sessions: {sessions.length}</span>
            <span>Current: {currentSessionId.split('_')[1] || 'New'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
