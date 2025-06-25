#!/usr/bin/env python3
"""
Test script with multiple documents to see more status transitions
"""
import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1/sec"

def test_multiple_documents():
    """Test with multiple documents to see more status changes"""
    print("🔄 Testing Multiple Document Batch Processing...")
    print("=" * 60)
    
    # Start batch processing with multiple documents
    batch_request = {
        "filings": [
            {
                "ticker": "AAPL",
                "accession_number": "0000320193-23-000077",
                "document_id": None
            },
            {
                "ticker": "MSFT", 
                "accession_number": "0000789019-23-000095",
                "document_id": None
            }
        ],
        "max_parallel": 1,  # Process one at a time to see more transitions
        "batch_id": f"multi_test_{int(time.time())}"
    }
    
    print(f"🚀 Starting batch with {len(batch_request['filings'])} documents...")
    print(f"   Batch ID: {batch_request['batch_id']}")
    
    try:
        # Start the batch
        response = requests.post(f"{BASE_URL}/documents/process-multiple", 
                               json=batch_request, 
                               timeout=3)
        
        if response.status_code == 200:
            result = response.json()
            batch_id = result.get("batch_id")
            print(f"✅ Batch submitted! ID: {batch_id}")
        else:
            print(f"❌ Failed to start batch: {response.status_code}")
            return
            
    except requests.exceptions.Timeout:
        batch_id = batch_request['batch_id']
        print(f"⏱️  Request timed out (expected) - monitoring batch: {batch_id}")
    except Exception as e:
        print(f"❌ Error starting batch: {e}")
        return
    
    # Monitor with more frequent polling
    print(f"\n🔍 Monitoring batch status...")
    print("Time     | Status      | Progress | Completed | Failed | Current Processing")
    print("-" * 85)
    
    for i in range(60):  # Monitor for up to 120 seconds
        try:
            status_response = requests.get(f"{BASE_URL}/batch/{batch_id}/status", timeout=2)
            
            if status_response.status_code == 200:
                status = status_response.json()
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Show detailed processing info
                processing_details = []
                for doc in status.get('current_processing', []):
                    stage = doc.get('stage', 'unknown')
                    percent = doc.get('progress_percent', 0)
                    doc_id = doc.get('document_id', 'unknown')
                    if len(doc_id) > 15:
                        doc_id = doc_id[:12] + "..."
                    processing_details.append(f"{doc_id}:{stage}({percent:.0f}%)")
                
                processing_str = ', '.join(processing_details) if processing_details else "None"
                
                print(f"{current_time} | {status.get('status', 'unknown'):11} | {status.get('overall_progress_percent', 0):6.1f}% | {status.get('completed_documents', 0):9} | {status.get('failed_documents', 0):6} | {processing_str}")
                
                # Check if completed
                if status.get('overall_progress_percent', 0) >= 100:
                    print(f"\n✅ Processing completed!")
                    break
                    
            elif status_response.status_code == 404:
                print(f"{datetime.now().strftime('%H:%M:%S')} | ❌ BATCH NOT FOUND!")
                break
            else:
                print(f"{datetime.now().strftime('%H:%M:%S')} | ⚠️  Status error: {status_response.status_code}")
                
        except Exception as e:
            print(f"{datetime.now().strftime('%H:%M:%S')} | ⚠️  Error: {e}")
        
        time.sleep(1)  # Check every second for more detail
    
    print("\n🏁 Multi-document test completed!")

if __name__ == "__main__":
    test_multiple_documents()
