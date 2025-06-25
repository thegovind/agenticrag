#!/usr/bin/env python3
"""
Test script to monitor batch processing status in real-time
"""
import requests
import time
import json
import asyncio
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1/sec"

def test_real_time_batch_updates():
    """Test real-time batch processing status updates"""
    print("🔄 Testing Real-time Batch Status Updates...")
    print("=" * 60)
    
    # Start batch processing
    batch_request = {
        "filings": [
            {
                "ticker": "AAPL",
                "accession_number": "0000320193-23-000077",  # Apple's recent 10-K
                "document_id": None
            }
        ],
        "max_parallel": 1,
        "batch_id": f"realtime_test_{int(time.time())}"
    }
    
    print(f"🚀 Starting batch processing...")
    print(f"   Batch ID: {batch_request['batch_id']}")
    print(f"   Documents: {len(batch_request['filings'])}")
    
    try:
        # Start the batch (don't wait for completion)
        response = requests.post(f"{BASE_URL}/documents/process-multiple", 
                               json=batch_request, 
                               timeout=5)  # Short timeout
        
        if response.status_code == 200:
            result = response.json()
            batch_id = result.get("batch_id")
            print(f"✅ Batch submitted! ID: {batch_id}")
        else:
            print(f"❌ Failed to start batch: {response.status_code}")
            return
            
    except requests.exceptions.Timeout:
        # This is expected - the request will timeout while processing
        batch_id = batch_request['batch_id']
        print(f"⏱️  Request timed out (expected) - monitoring batch: {batch_id}")
    except Exception as e:
        print(f"❌ Error starting batch: {e}")
        return
    
    # Monitor status in real-time
    print(f"\n🔍 Monitoring batch status for {batch_id}...")
    print("Time     | Status      | Progress | Processing Documents")
    print("-" * 70)
    
    last_status = None
    for i in range(30):  # Monitor for up to 60 seconds
        try:
            status_response = requests.get(f"{BASE_URL}/batch/{batch_id}/status", timeout=2)
            
            if status_response.status_code == 200:
                status = status_response.json()
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Only print if status changed
                current_status_summary = {
                    'status': status.get('status'),
                    'progress': status.get('overall_progress_percent'),
                    'processing_count': len(status.get('current_processing', []))
                }
                
                if current_status_summary != last_status:
                    processing_docs = []
                    for doc in status.get('current_processing', []):
                        stage = doc.get('stage', 'unknown')
                        percent = doc.get('progress_percent', 0)
                        processing_docs.append(f"{doc.get('document_id', 'unknown')}:{stage}({percent:.0f}%)")
                    
                    processing_str = ', '.join(processing_docs) if processing_docs else "None"
                    if len(processing_str) > 30:
                        processing_str = processing_str[:27] + "..."
                    
                    print(f"{current_time} | {status.get('status', 'unknown'):11} | {status.get('overall_progress_percent', 0):6.1f}% | {processing_str}")
                    last_status = current_status_summary
                
                # Check if completed
                if status.get('overall_progress_percent', 0) >= 100:
                    print(f"\n✅ Processing completed!")
                    print(f"   Completed: {status.get('completed_documents', 0)}")
                    print(f"   Failed: {status.get('failed_documents', 0)}")
                    if status.get('finished_at'):
                        print(f"   Finished at: {status.get('finished_at')}")
                    break
                    
            elif status_response.status_code == 404:
                print(f"{datetime.now().strftime('%H:%M:%S')} | ❌ BATCH NOT FOUND - Status was deleted!")
                break
            else:
                print(f"{datetime.now().strftime('%H:%M:%S')} | ⚠️  Status check failed: {status_response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"{datetime.now().strftime('%H:%M:%S')} | ⚠️  Request error: {e}")
        
        time.sleep(2)  # Check every 2 seconds
    
    print("\n" + "=" * 60)
    print("🏁 Real-time monitoring completed!")

if __name__ == "__main__":
    test_real_time_batch_updates()
