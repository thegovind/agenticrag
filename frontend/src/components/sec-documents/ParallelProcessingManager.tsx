import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { 
  Play, 
  Square, 
  Clock, 
  CheckCircle, 
  XCircle, 
  Loader2,
  FileText,
  Zap
} from 'lucide-react';
import { apiService } from '@/services/api';

interface BatchStatus {
  batch_id: string;
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  current_processing: Array<{
    document_id: string;
    ticker: string;
    accession_number: string;
    stage: string;
    progress_percent: number;
    message: string;
    started_at: string;
    updated_at: string;
    completed_at?: string;
    error_message?: string;
    chunks_created: number;
    tokens_used: number;
  }>;
  overall_progress_percent: number;
  started_at: string;
  estimated_completion?: string;
}

interface ProcessDocumentRequest {
  ticker: string;
  accession_number: string;
  document_id?: string;
}

const ParallelProcessingManager: React.FC = () => {  const [documents, setDocuments] = useState<ProcessDocumentRequest[]>([]);
  const [newTicker, setNewTicker] = useState('');
  const [newAccessionNumber, setNewAccessionNumber] = useState('');
  const [maxParallel, setMaxParallel] = useState(3);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const [processedResults, setProcessedResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const addDocument = () => {
    if (newTicker && newAccessionNumber) {
      setDocuments(prev => [...prev, {
        ticker: newTicker.toUpperCase(),
        accession_number: newAccessionNumber,
        document_id: undefined
      }]);
      setNewTicker('');
      setNewAccessionNumber('');
    }
  };

  const removeDocument = (index: number) => {
    setDocuments(prev => prev.filter((_, i) => i !== index));
  };

  const startProcessing = async () => {
    if (documents.length === 0) return;

    setIsProcessing(true);
    setError(null);
    setProcessedResults([]);    try {
      console.log('Starting processing of', documents.length, 'documents');
      const response = await apiService.processMultipleSECDocuments({
        filings: documents,
        max_parallel: maxParallel
      });
      
      console.log('Processing response:', response);
      setProcessedResults(response.results);
      
      // Start monitoring progress
      startStatusChecking(response.batch_id);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start processing');
      setIsProcessing(false);
    }
  };
  const startStatusChecking = (batchId: string) => {
    console.log('Starting status checking for batch:', batchId);
    
    // Check immediately first
    const checkStatus = async () => {
      try {
        console.log('Checking batch status...');
        const response = await apiService.getBatchStatus(batchId);
        const status: BatchStatus = response;
        console.log('Batch status update:', status);
        setBatchStatus(status);
        
        // Stop checking if processing is complete
        if (status.overall_progress_percent >= 100) {
          console.log('Processing complete, stopping status checks');
          setIsProcessing(false);
          if (statusCheckInterval.current) {
            clearInterval(statusCheckInterval.current);
            statusCheckInterval.current = null;
          }
          return false; // Stop checking
        }
        return true; // Continue checking
      } catch (err: any) {
        console.error('Failed to check status:', err);
        
        // If it's a 404, the batch might not be created yet or was cleaned up
        if (err.message && err.message.includes('404')) {
          console.log('Batch not found (404) - will retry...');
          return true; // Continue checking, batch might not be created yet
        }
        
        // For other errors, continue checking for a while
        return true;
      }
    };
    
    // Check immediately
    checkStatus();
    
    // Then check every 500ms for more responsive updates
    statusCheckInterval.current = setInterval(async () => {
      const shouldContinue = await checkStatus();
      if (!shouldContinue && statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
        statusCheckInterval.current = null;
      }
    }, 500); // Reduced from 2000ms to 500ms for better responsiveness
  };

  const stopProcessing = () => {    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
      statusCheckInterval.current = null;
    }
    setIsProcessing(false);
    setBatchStatus(null);
  };

  useEffect(() => {
    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
    };
  }, []);

  const getStageIcon = (stage: string) => {
    switch (stage.toUpperCase()) {
      case 'COMPLETED': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'FAILED': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'QUEUED': return <Clock className="h-4 w-4 text-gray-500" />;
      case 'DOWNLOADING': return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'PARSING': return <Loader2 className="h-4 w-4 text-yellow-500 animate-spin" />;
      case 'CHUNKING': return <Loader2 className="h-4 w-4 text-orange-500 animate-spin" />;
      case 'EMBEDDING': return <Loader2 className="h-4 w-4 text-purple-500 animate-spin" />;
      case 'INDEXING': return <Loader2 className="h-4 w-4 text-indigo-500 animate-spin" />;
      default: return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
    }
  };
  
  const getStageColor = (stage: string) => {
    switch (stage.toUpperCase()) {
      case 'COMPLETED': return 'bg-green-100 text-green-800';
      case 'FAILED': return 'bg-red-100 text-red-800';
      case 'QUEUED': return 'bg-gray-100 text-gray-800';
      case 'DOWNLOADING': return 'bg-blue-100 text-blue-800';
      case 'PARSING': return 'bg-yellow-100 text-yellow-800';
      case 'CHUNKING': return 'bg-orange-100 text-orange-800';
      case 'EMBEDDING': return 'bg-purple-100 text-purple-800';
      case 'INDEXING': return 'bg-indigo-100 text-indigo-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Parallel SEC Document Processing
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Process multiple SEC documents in parallel for faster bulk operations
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="ticker">Ticker Symbol</Label>
              <Input
                id="ticker"
                value={newTicker}
                onChange={(e) => setNewTicker(e.target.value)}
                placeholder="AAPL"
                disabled={isProcessing}
              />
            </div>
            <div>
              <Label htmlFor="accession">Accession Number</Label>
              <Input
                id="accession"
                value={newAccessionNumber}
                onChange={(e) => setNewAccessionNumber(e.target.value)}
                placeholder="0000320193-24-000123"
                disabled={isProcessing}
              />
            </div>
            <div>
              <Label htmlFor="parallel">Max Parallel</Label>
              <Input
                id="parallel"
                type="number"
                min="1"
                max="10"
                value={maxParallel}
                onChange={(e) => setMaxParallel(parseInt(e.target.value) || 3)}
                disabled={isProcessing}
              />
            </div>
            <div className="flex items-end">
              <Button onClick={addDocument} disabled={isProcessing} className="w-full">
                Add Document
              </Button>
            </div>
          </div>

          {/* Document Queue */}
          {documents.length > 0 && (
            <div>
              <h3 className="text-sm font-medium mb-2">Document Queue ({documents.length})</h3>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {documents.map((doc, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">{doc.ticker} - {doc.accession_number}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeDocument(index)}
                      disabled={isProcessing}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="flex gap-2">
            <Button
              onClick={startProcessing}
              disabled={documents.length === 0 || isProcessing}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              Start Processing
            </Button>
            <Button
              variant="outline"
              onClick={stopProcessing}
              disabled={!isProcessing}
              className="flex items-center gap-2"
            >
              <Square className="h-4 w-4" />
              Stop
            </Button>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
              {error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Processing Status */}
      {batchStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Processing Status</span>
              <Badge variant={isProcessing ? "default" : "secondary"}>
                {isProcessing ? "Processing" : "Completed"}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Overall Progress */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Overall Progress</span>
                <span>{Math.round(batchStatus.overall_progress_percent)}%</span>
              </div>
              <Progress value={batchStatus.overall_progress_percent} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Completed: {batchStatus.completed_documents}</span>
                <span>Failed: {batchStatus.failed_documents}</span>
                <span>Total: {batchStatus.total_documents}</span>
              </div>
            </div>

            <Separator />            {/* Current Processing */}
            {batchStatus.current_processing.length > 0 ? (
              <div>
                <h3 className="text-sm font-medium mb-3">Currently Processing</h3>
                <div className="space-y-3">
                  {batchStatus.current_processing.map((doc, index) => (
                    <div key={`${doc.ticker}-${index}`} className="border rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getStageIcon(doc.stage)}
                          <span className="font-medium text-sm">
                            {doc.ticker} - {doc.accession_number}
                          </span>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${getStageColor(doc.stage)}`}>
                            {doc.stage.toUpperCase()}
                          </span>
                        </div>
                        <span className="text-sm text-muted-foreground">
                          {Math.round(doc.progress_percent)}%
                        </span>
                      </div>
                      <Progress value={doc.progress_percent} className="h-1 mb-2" />
                      {doc.error_message && (
                        <p className="text-xs text-red-500">{doc.error_message}</p>
                      )}
                      <p className="text-xs text-muted-foreground">{doc.message}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : batchStatus.overall_progress_percent < 100 && isProcessing ? (
              <div className="text-center py-4">
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing documents...</span>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {processedResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {processedResults.map((result, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded">
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4" />
                    <span className="font-medium">{result.document_id}</span>
                    {result.skipped && <Badge variant="secondary">Skipped</Badge>}
                  </div>
                  <div className="text-right text-sm text-muted-foreground">
                    {result.chunks_created} chunks • {result.tokens_used || 0} tokens
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ParallelProcessingManager;
