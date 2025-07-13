from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class SessionMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class SessionInfo(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    title: Optional[str] = None
    messages: List[SessionMessage] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True

    class Config:
        arbitrary_types_allowed = True

class SessionHistory(BaseModel):
    sessions: List[SessionInfo] = Field(default_factory=list)
    total_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True

def validate_session_data(session_data: Dict[str, Any]) -> SessionInfo:
    """Validate and convert dictionary session data to SessionInfo model"""
    try:
        if "messages" in session_data:
            messages = []
            for msg in session_data["messages"]:
                if isinstance(msg, dict):
                    if "role" not in msg or "content" not in msg:
                        continue
                    messages.append(SessionMessage(**msg))
                else:
                    messages.append(msg)
            session_data["messages"] = messages
        
        if "session_id" not in session_data:
            session_data["session_id"] = str(uuid.uuid4())
        
        for field in ["created_at", "updated_at"]:
            if field in session_data and isinstance(session_data[field], str):
                try:
                    session_data[field] = datetime.fromisoformat(session_data[field].replace('Z', '+00:00'))
                except ValueError:
                    session_data[field] = datetime.utcnow()
        
        return SessionInfo(**session_data)
    except Exception as e:
        raise ValueError(f"Invalid session data structure: {str(e)}")

def serialize_session_for_storage(session: SessionInfo) -> Dict[str, Any]:
    """Serialize session data for storage in CosmosDB or other systems"""
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "title": session.title,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in session.messages
        ],
        "metadata": session.metadata,
        "is_active": session.is_active
    }
