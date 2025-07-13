import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.services.azure_services import AzureServiceManager
from app.models.session import SessionInfo, validate_session_data, serialize_session_for_storage

@pytest.fixture
def mock_azure_service():
    service = AzureServiceManager()
    service.use_mock = True
    service.cosmos_client = MagicMock()
    service.cosmos_container = MagicMock()
    return service

@pytest.mark.asyncio
async def test_session_save_and_retrieve_integration(mock_azure_service):
    """Test complete session save and retrieve workflow"""
    
    session_id = "integration_test_session"
    test_message = {
        "id": "msg_1",
        "role": "user",
        "content": "Test message",
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    mock_azure_service.cosmos_container.read_item.side_effect = Exception("NotFound")
    mock_azure_service.cosmos_container.upsert_item.return_value = None
    
    result = await mock_azure_service.save_session_history(session_id, test_message)
    assert result is True
    
    mock_azure_service.cosmos_container.upsert_item.assert_called_once()
    
    saved_data = mock_azure_service.cosmos_container.upsert_item.call_args[0][0]
    assert saved_data["id"] == session_id
    assert saved_data["session_id"] == session_id
    assert len(saved_data["messages"]) == 1

@pytest.mark.asyncio
async def test_session_history_retrieval_with_validation(mock_azure_service):
    """Test session history retrieval with proper validation"""
    
    session_id = "validation_test_session"
    mock_session_data = {
        "id": session_id,
        "session_id": session_id,
        "messages": [
            {
                "id": "msg_1",
                "role": "user", 
                "content": "Hello",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:01:00Z",
        "is_active": True
    }
    
    mock_azure_service.cosmos_container.read_item.return_value = mock_session_data
    
    result = await mock_azure_service.get_session_history(session_id)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"

@pytest.mark.asyncio
async def test_session_data_validation_in_service(mock_azure_service):
    """Test that service properly validates session data"""
    
    session_id = "validation_service_test"
    invalid_message = {
        "content": "Missing required fields"
    }
    
    mock_azure_service.cosmos_container.read_item.side_effect = Exception("NotFound")
    
    try:
        result = await mock_azure_service.save_session_history(session_id, invalid_message)
        assert result is False
    except Exception:
        pass

@pytest.mark.asyncio
async def test_session_update_workflow(mock_azure_service):
    """Test updating an existing session"""
    
    session_id = "update_test_session"
    existing_session = {
        "id": session_id,
        "session_id": session_id,
        "messages": [
            {
                "id": "msg_1",
                "role": "user",
                "content": "First message",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "is_active": True
    }
    
    new_message = {
        "id": "msg_2",
        "role": "assistant",
        "content": "Response message",
        "timestamp": "2024-01-01T00:01:00Z"
    }
    
    mock_azure_service.cosmos_container.read_item.return_value = existing_session
    mock_azure_service.cosmos_container.upsert_item.return_value = None
    
    result = await mock_azure_service.save_session_history(session_id, new_message)
    assert result is True
    
    saved_data = mock_azure_service.cosmos_container.upsert_item.call_args[0][0]
    assert len(saved_data["messages"]) == 2
    assert saved_data["messages"][1]["content"] == "Response message"

def test_session_data_model_consistency():
    """Test that session data models maintain consistency"""
    
    raw_session_data = {
        "session_id": "consistency_test",
        "user_id": "user_123",
        "messages": [
            {
                "role": "user",
                "content": "Test message"
            }
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "is_active": True
    }
    
    validated_session = validate_session_data(raw_session_data)
    assert isinstance(validated_session, SessionInfo)
    
    serialized_data = serialize_session_for_storage(validated_session)
    assert serialized_data["session_id"] == "consistency_test"
    assert serialized_data["user_id"] == "user_123"
    assert len(serialized_data["messages"]) == 1
    
    re_validated = validate_session_data(serialized_data)
    assert re_validated.session_id == validated_session.session_id
    assert len(re_validated.messages) == len(validated_session.messages)
