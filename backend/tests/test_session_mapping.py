import pytest
import json
from datetime import datetime
from app.models.session import SessionInfo, SessionMessage, validate_session_data, serialize_session_for_storage

def test_session_message_validation():
    """Test SessionMessage model validation"""
    message_data = {
        "role": "user",
        "content": "Test message",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    message = SessionMessage(**message_data)
    assert message.role == "user"
    assert message.content == "Test message"
    assert message.id is not None  # Should auto-generate

def test_session_info_validation():
    """Test SessionInfo model validation"""
    session_data = {
        "session_id": "test-session-123",
        "user_id": "user-456",
        "title": "Test Session",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "is_active": True
    }
    
    session = SessionInfo(**session_data)
    assert session.session_id == "test-session-123"
    assert session.user_id == "user-456"
    assert session.title == "Test Session"
    assert len(session.messages) == 1
    assert isinstance(session.messages[0], SessionMessage)

def test_validate_session_data():
    """Test session data validation function"""
    raw_data = {
        "session_id": "test-123",
        "messages": [
            {
                "role": "user",
                "content": "Test message"
            }
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z"
    }
    
    session = validate_session_data(raw_data)
    assert isinstance(session, SessionInfo)
    assert session.session_id == "test-123"
    assert len(session.messages) == 1

def test_validate_session_data_missing_session_id():
    """Test validation generates session_id if missing"""
    raw_data = {
        "messages": [],
        "is_active": True
    }
    
    session = validate_session_data(raw_data)
    assert session.session_id is not None
    assert len(session.session_id) > 0

def test_validate_session_data_invalid_messages():
    """Test validation handles invalid messages gracefully"""
    raw_data = {
        "session_id": "test-123",
        "messages": [
            {"role": "user", "content": "Valid message"},
            {"invalid": "message"},  # Missing required fields
            {"role": "assistant", "content": "Another valid message"}
        ]
    }
    
    session = validate_session_data(raw_data)
    assert len(session.messages) == 2  # Invalid message should be filtered out

def test_serialize_session_for_storage():
    """Test session serialization for storage"""
    session = SessionInfo(
        session_id="test-123",
        user_id="user-456",
        title="Test Session",
        messages=[
            SessionMessage(role="user", content="Hello"),
            SessionMessage(role="assistant", content="Hi there!")
        ]
    )
    
    serialized = serialize_session_for_storage(session)
    
    assert serialized["session_id"] == "test-123"
    assert serialized["user_id"] == "user-456"
    assert serialized["title"] == "Test Session"
    assert len(serialized["messages"]) == 2
    assert all(isinstance(msg["timestamp"], str) for msg in serialized["messages"])
    assert serialized["is_active"] is True

def test_session_data_consistency():
    """Test that session data maintains consistency through validation and serialization"""
    original_data = {
        "session_id": "consistency-test",
        "user_id": "user-123",
        "title": "Consistency Test",
        "messages": [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Response 1"}
        ],
        "metadata": {"test": "value"},
        "is_active": True
    }
    
    session = validate_session_data(original_data)
    
    serialized = serialize_session_for_storage(session)
    
    session2 = validate_session_data(serialized)
    
    assert session.session_id == session2.session_id
    assert session.user_id == session2.user_id
    assert session.title == session2.title
    assert len(session.messages) == len(session2.messages)
    assert session.is_active == session2.is_active

def test_datetime_handling():
    """Test proper datetime handling in session data"""
    session_data = {
        "session_id": "datetime-test",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00.123456Z",
        "messages": []
    }
    
    session = validate_session_data(session_data)
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)
    
    serialized = serialize_session_for_storage(session)
    assert isinstance(serialized["created_at"], str)
    assert isinstance(serialized["updated_at"], str)
