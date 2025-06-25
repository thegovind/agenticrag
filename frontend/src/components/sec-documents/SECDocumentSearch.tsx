import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Search, Download, FileText, Calendar, ExternalLink, Loader2, Clock, CheckCircle, XCircle, AlertCircle, Square } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';

interface Filing {
  ticker: string;
  company_name: string;
  cik: string;
  form_type: string;
  filing_date: string;
  period_end_date?: string;
  document_url: string;
  accession_number: string;
  file_size?: number;
  year: number;
}

interface SearchFilters {
  ticker: string;
  formTypes: string[];
  years: number[];
}

interface DocumentProcessingProgress {
  document_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress_percent: number;
  started_at: string;
  updated_at: string;
  completed_at?: string;
  error_message?: string;
  chunks_created: number;
  tokens_used: number;
}

interface BatchProcessingStatus {
  batch_id: string;
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  current_processing: DocumentProcessingProgress[];
  overall_progress_percent: number;
  started_at: string;
  finished_at?: string;
  estimated_completion?: string;
  status: string; // "processing", "completed", "failed"
  error_message?: string;
}

const SECDocumentSearch: React.FC = () => {
  // Search form state
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    ticker: '',
    formTypes: ['10-K'],
    years: [new Date().getFullYear()]
  });
  
  // Results and processing state
  const [filings, setFilings] = useState<Filing[]>([]);
  const [selectedFilings, setSelectedFilings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  // Batch processing state
  const [batchStatus, setBatchStatus] = useState<BatchProcessingStatus | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null); // Use ref to avoid stale closures
  const pollingCountRef = useRef<number>(0); // Counter to track polling attempts
  
  const { toast } = useToast();

  const availableFormTypes = [
    { value: '10-K', label: '10-K (Annual Report)' },
    { value: '10-Q', label: '10-Q (Quarterly Report)' },
    { value: '8-K', label: '8-K (Current Report)' },
    { value: '10-K/A', label: '10-K/A (Annual Report Amendment)' },
    { value: '10-Q/A', label: '10-Q/A (Quarterly Report Amendment)' },
    { value: '20-F', label: '20-F (Foreign Annual)' },
    { value: 'DEF 14A', label: 'DEF 14A (Proxy Statement)' }
  ];

  const currentYear = new Date().getFullYear();
  const availableYears = Array.from({ length: 10 }, (_, i) => currentYear - i);

  // Add cleanup effect for polling interval - FIXED: removed dependency array
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        console.log('🧹 Component cleanup: clearing polling interval');
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []); // REMOVED progressPollingInterval from dependency array to prevent race condition
  // Simple and reliable progress polling function with aggressive debugging
  // Use useCallback to prevent stale closures and unnecessary recreations
  const pollBatchProgress = useCallback(async (batchId: string) => {
    try {
      pollingCountRef.current += 1;
      const timestamp = new Date().toLocaleTimeString();
      console.log(`🔄 [${timestamp}] POLLING ATTEMPT #${pollingCountRef.current} for batch:`, batchId);
      console.log(`🔄 [${timestamp}] Current polling interval exists:`, pollingIntervalRef.current !== null);
      
      const status = await apiService.getBatchStatus(batchId);
      console.log(`✅ [${timestamp}] RECEIVED batch status:`, {
        progress: status.overall_progress_percent,
        completed: status.completed_documents,
        failed: status.failed_documents,
        processing: status.current_processing?.length || 0,
        batch_status: status.status
      });
      
      // Always update the status - let React handle re-render optimization
      setBatchStatus(status as any);
      
      // Stop polling if batch is complete OR if status is "completed"
      if (status.overall_progress_percent >= 100 || status.status === "completed") {
        console.log(`🏁 [${timestamp}] BATCH COMPLETE (${status.overall_progress_percent}% / status: ${status.status}), stopping polling`);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
          console.log(`🛑 [${timestamp}] Polling interval CLEARED`);
        }
        
        // Show completion toast
        const completedCount = status.completed_documents;
        const failedCount = status.failed_documents;
        
        if (failedCount > 0) {
          toast({
            title: "Processing Completed with Errors",
            description: `${completedCount} documents processed successfully, ${failedCount} failed.`,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Processing Complete! 🎉",
            description: `Successfully processed ${completedCount} document(s) and added them to the knowledge base.`,
          });
        }
        
        // Clear processing state after 3 seconds
        setTimeout(() => {
          setProcessing(false);
          setBatchStatus(null);
          console.log(`🧹 [${new Date().toLocaleTimeString()}] Processing state CLEARED`);
        }, 3000);
      } else {
        console.log(`⏳ [${timestamp}] Still processing... ${status.overall_progress_percent}% complete`);
      }
    } catch (error) {
      console.error('❌ ERROR polling batch progress:', error);
      console.error('❌ Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : 'No stack trace'
      });
      
      // Check if this is a "batch not found" error (404)
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('404') || errorMessage.includes('not found')) {
        console.log('⏳ Batch not found yet, will retry on next poll (API call may still be in progress)');
        // Don't stop polling for 404 errors - the backend might not have created the batch yet
        return;
      }
      
      // On other errors, stop polling
      if (pollingIntervalRef.current) {
        console.log('🛑 Stopping polling due to error');
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      setProcessing(false);
    }
  }, [toast]); // Only depend on toast, not on state variables

  // Separate function to start polling - also use useCallback to prevent recreation
  const startPolling = useCallback((batchId: string) => {
    console.log('🚀 Starting polling for batch:', batchId);
    
    // Reset polling counter
    pollingCountRef.current = 0;
    
    // Clear any existing interval first
    if (pollingIntervalRef.current) {
      console.log('🧹 Clearing existing interval before starting new one');
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    // Poll once immediately
    console.log('📞 Calling initial poll...');
    pollBatchProgress(batchId);
    
    // Then set up interval polling every 500ms for real-time feedback
    console.log('⏰ Setting up 500ms interval...');
    const interval = setInterval(() => {
      console.log('📞 Interval tick - calling pollBatchProgress');
      pollBatchProgress(batchId);
    }, 500); // Changed from 2000ms to 500ms for real-time updates
    
    pollingIntervalRef.current = interval;
    console.log('✅ Polling interval set up with ID:', interval);
    console.log('✅ Interval stored in ref');
  }, [pollBatchProgress]); // Depend on pollBatchProgress

  const handleFormTypeChange = (formType: string, checked: boolean) => {
    setSearchFilters(prev => ({
      ...prev,
      formTypes: checked 
        ? [...prev.formTypes, formType]
        : prev.formTypes.filter(t => t !== formType)
    }));
  };

  const handleFilingSelection = (accessionNumber: string, checked: boolean) => {
    setSelectedFilings(prev => 
      checked 
        ? [...prev, accessionNumber]
        : prev.filter(id => id !== accessionNumber)
    );
  };

  const handleYearSelection = (year: number, checked: boolean) => {
    setSearchFilters(prev => ({
      ...prev,
      years: checked 
        ? [...prev.years, year]
        : prev.years.filter(y => y !== year)
    }));
  };

  const searchFilings = async () => {
    if (!searchFilters.ticker.trim()) {
      toast({
        title: "Search Error",
        description: "Please enter a company ticker or name.",
        variant: "destructive",
      });
      return;
    }

    if (searchFilters.formTypes.length === 0) {
      toast({
        title: "Search Error", 
        description: "Please select at least one document type.",
        variant: "destructive",
      });
      return;
    }

    if (searchFilters.years.length === 0) {
      toast({
        title: "Search Error", 
        description: "Please select at least one year.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setFilings([]);
    setSelectedFilings([]);

    try {
      const response = await fetch('/api/v1/sec/filings/specific', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: searchFilters.ticker.toUpperCase(),
          form_types: searchFilters.formTypes,
          years: searchFilters.years,
          limit: 50
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch filings');
      }

      const data = await response.json();
      setFilings(data.filings || []);

      toast({
        title: "Search Complete",
        description: `Found ${data.filings?.length || 0} filings for ${searchFilters.ticker}`,
      });

    } catch (error) {
      console.error('Search error:', error);
      toast({
        title: "Search Error",
        description: "Failed to search SEC filings. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };
  const processSelectedFilings = async () => {
    if (selectedFilings.length === 0) {
      toast({
        title: "Processing Error",
        description: "Please select at least one filing to process.",
        variant: "destructive",
      });
      return;
    }

    setProcessing(true);

    try {
      // Prepare batch request
      const filingsToProcess = selectedFilings.map(accessionNumber => {
        const filing = filings.find(f => f.accession_number === accessionNumber);
        return {
          ticker: filing?.ticker || '',
          accession_number: accessionNumber
        };
      }).filter(f => f.ticker);

      // Start batch processing - DON'T WAIT for completion, start polling immediately
      console.log('🚀 Initiating batch processing request...');
      
      // Generate batch ID immediately for polling
      const batchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Start the API call but don't await it - let it run in background
      apiService.processMultipleSECDocuments({
        filings: filingsToProcess,
        max_parallel: 10,
        batch_id: batchId  // Send our generated batch_id
      }).then(response => {
        console.log('✅ Batch processing API call completed:', response);
        // API call is done, but polling should already be running
      }).catch(error => {
        console.error('❌ Batch processing API call failed:', error);
        // If API call fails, stop polling and show error
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setProcessing(false);
        setBatchStatus(null);
        toast({
          title: "Processing Error",
          description: "Failed to start processing. Please try again.",
          variant: "destructive",
        });
      });

      // Set initial batch status with our generated batch_id
      setBatchStatus({
        batch_id: batchId,
        total_documents: selectedFilings.length,
        completed_documents: 0,
        failed_documents: 0,
        current_processing: [],
        overall_progress_percent: 0,
        started_at: new Date().toISOString(),
        status: "queued"  // Start as queued
      });

      // Start polling immediately without waiting for API call
      console.log('🚀 Starting polling immediately with batch_id:', batchId);
      setTimeout(() => {
        startPolling(batchId);
      }, 100); // Small delay to let React settle

      toast({
        title: "Processing Started",
        description: `Started processing ${selectedFilings.length} document(s) in parallel. Watch console for real-time updates.`,
      });

    } catch (error) {
      console.error('Processing error:', error);
      toast({
        title: "Processing Error",
        description: "Failed to start processing. Please try again.",
        variant: "destructive",
      });
      setProcessing(false);
    }
  };

  // Helper function to get document progress for a specific filing
  const getDocumentProgress = (filing: Filing): DocumentProcessingProgress | null => {
    if (!batchStatus?.current_processing) return null;
    
    const documentId = `${filing.ticker}_${filing.accession_number}`;
    return batchStatus.current_processing.find(doc => doc.document_id === documentId) || null;
  };

  // Helper function to determine if a filing is currently selected for processing
  const isFilingBeingProcessed = (filing: Filing): boolean => {
    return processing && selectedFilings.includes(filing.accession_number);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            SEC Document Search & Process
          </CardTitle>
          <CardDescription>
            Search for SEC filings by company, document type, and year, then process them in parallel
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Company Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Company Ticker or Name</label>
            <Input
              placeholder="e.g., AAPL, Apple Inc."
              value={searchFilters.ticker}
              onChange={(e) => setSearchFilters(prev => ({ ...prev, ticker: e.target.value }))}
              onKeyDown={(e) => e.key === 'Enter' && searchFilings()}
            />
          </div>

          {/* Document Types */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Document Types</label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {availableFormTypes.map((formType) => (
                <div key={formType.value} className="flex items-center space-x-2">
                  <Checkbox
                    id={formType.value}
                    checked={searchFilters.formTypes.includes(formType.value)}
                    onCheckedChange={(checked) => 
                      handleFormTypeChange(formType.value, checked as boolean)
                    }
                  />
                  <label 
                    htmlFor={formType.value} 
                    className="text-sm leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {formType.label}
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Years Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Years</label>
            <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
              {availableYears.map((year) => (
                <div key={year} className="flex items-center space-x-2">
                  <Checkbox
                    id={`year-${year}`}
                    checked={searchFilters.years.includes(year)}
                    onCheckedChange={(checked) => 
                      handleYearSelection(year, checked as boolean)
                    }
                  />
                  <label 
                    htmlFor={`year-${year}`} 
                    className="text-sm leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {year}
                  </label>
                </div>
              ))}
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSearchFilters(prev => ({ 
                  ...prev, 
                  years: availableYears 
                }))}
              >
                Select All Years
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSearchFilters(prev => ({ 
                  ...prev, 
                  years: [] 
                }))}
              >
                Clear All Years
              </Button>
            </div>
          </div>

          {/* Search Button */}
          <Button 
            onClick={searchFilings}
            disabled={loading || !searchFilters.ticker.trim() || searchFilters.formTypes.length === 0 || searchFilters.years.length === 0}
            className="w-full md:w-auto"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <Search className="h-4 w-4 mr-2" />
                Search SEC Filings
              </>
            )}
          </Button>
        </CardContent>
      </Card>      {/* Batch Processing Progress */}
      {batchStatus && processing && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {batchStatus.overall_progress_percent >= 100 ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <Loader2 className="h-5 w-5 animate-spin" />
              )}
              {batchStatus.overall_progress_percent >= 100 ? 'Processing Complete!' : 'Batch Processing Progress'}
            </CardTitle>
            <CardDescription>
              Batch ID: {batchStatus.batch_id} • Started: {new Date(batchStatus.started_at).toLocaleTimeString()}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Overall Progress */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Overall Progress</span>
                <span>{Math.round(batchStatus.overall_progress_percent)}%</span>
              </div>
              <Progress value={batchStatus.overall_progress_percent} className="w-full" />
              <div className="flex justify-between text-xs text-gray-500">
                <span>✅ Completed: {batchStatus.completed_documents}</span>
                <span>❌ Failed: {batchStatus.failed_documents}</span>
                <span>📊 Total: {batchStatus.total_documents}</span>
              </div>
            </div>

            {/* Processing Summary */}
            {batchStatus.current_processing.length > 0 && batchStatus.overall_progress_percent < 100 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">
                  Currently Processing ({batchStatus.current_processing.length} document{batchStatus.current_processing.length !== 1 ? 's' : ''}):
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {batchStatus.current_processing.map((doc) => (
                    <div key={doc.document_id} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                      {getStatusIcon(doc.status)}
                      <span className="font-mono text-xs">{doc.document_id}</span>
                      <span className="ml-auto">{Math.round(doc.progress_percent)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {batchStatus.current_processing.length === 0 && batchStatus.overall_progress_percent < 100 && (
              <div className="text-sm text-gray-500 text-center py-2">
                📋 All documents are queued, processing will begin shortly...
              </div>
            )}

            {batchStatus.overall_progress_percent >= 100 && (
              <div className="text-sm text-center py-2 bg-green-50 border border-green-200 rounded">
                🎉 All selected documents have been processed and added to the knowledge base!
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* DEBUG: Test Interval Button */}
      {/* <Card className="border-dashed border-orange-200">
        <CardHeader>
          <CardTitle className="text-orange-600">🧪 Debug: Test Interval</CardTitle>
          <CardDescription>
            Test that JavaScript intervals work correctly in this environment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            onClick={() => {
              console.log('🧪 TEST: Starting test interval...');
              const testInterval = setInterval(() => {
                console.log('🧪 TEST: Interval tick at', new Date().toLocaleTimeString());
              }, 1000);
              
              setTimeout(() => {
                console.log('🧪 TEST: Stopping test interval');
                clearInterval(testInterval);
              }, 5000);
            }}
            variant="outline"
            className="border-orange-300 text-orange-600 hover:bg-orange-50"
          >
            Test 5-second Interval
          </Button>
        </CardContent>
      </Card> */}

      {/* Search Results */}
      {filings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results</CardTitle>
            <CardDescription>
              Found {filings.length} filing(s). Select the documents you want to process in parallel.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Select All/None and Process Button */}
              <div className="flex items-center justify-between border-b pb-4">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="select-all"
                    checked={selectedFilings.length === filings.length && filings.length > 0}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedFilings(filings.map(f => f.accession_number));
                      } else {
                        setSelectedFilings([]);
                      }
                    }}
                  />
                  <label htmlFor="select-all" className="text-sm font-medium">
                    Select All ({filings.length})
                  </label>
                </div>
                
                <div className="flex gap-2">
                  <Button
                    onClick={processSelectedFilings}
                    disabled={selectedFilings.length === 0 || processing}
                    variant="default"
                  >
                    {processing ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        Process Selected ({selectedFilings.length})
                      </>
                    )}
                  </Button>
                  
                  {processing && (
                    <Button
                      onClick={() => {
                        console.log('🛑 Stopping processing manually');
                        if (pollingIntervalRef.current) {
                          clearInterval(pollingIntervalRef.current);
                          pollingIntervalRef.current = null;
                        }
                        setProcessing(false);
                        setBatchStatus(null);
                      }}
                      variant="outline"
                      size="sm"
                    >
                      <Square className="h-4 w-4 mr-2" />
                      Stop
                    </Button>
                  )}
                  
                  {selectedFilings.length > 0 && !processing && (
                    <Button
                      onClick={() => setSelectedFilings([])}
                      variant="outline"
                      size="sm"
                    >
                      Clear Selection
                    </Button>
                  )}
                </div>
              </div>              {/* Filing List */}
              <div className="space-y-3">
                {filings.map((filing) => {
                  const documentProgress = getDocumentProgress(filing);
                  const isBeingProcessed = isFilingBeingProcessed(filing);
                  
                  return (
                    <Card key={filing.accession_number} className="hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Checkbox
                              id={filing.accession_number}
                              checked={selectedFilings.includes(filing.accession_number)}
                              onCheckedChange={(checked) => 
                                handleFilingSelection(filing.accession_number, checked as boolean)
                              }
                              disabled={processing}
                            />
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                <FileText className="h-4 w-4" />
                                <span className="font-semibold">{filing.form_type}</span>
                                <Badge variant="outline">{filing.ticker}</Badge>
                                <Badge variant="secondary">
                                  <Calendar className="h-3 w-3 mr-1" />
                                  {new Date(filing.filing_date).toLocaleDateString()}
                                </Badge>
                                
                                {/* Processing Status Badge */}
                                {isBeingProcessed && documentProgress && (
                                  <Badge 
                                    variant={
                                      documentProgress.status === 'completed' ? 'default' : 
                                      documentProgress.status === 'failed' ? 'destructive' : 
                                      'secondary'
                                    }
                                    className="flex items-center gap-1"
                                  >
                                    {getStatusIcon(documentProgress.status)}
                                    {documentProgress.status}
                                  </Badge>
                                )}
                                {isBeingProcessed && !documentProgress && (
                                  <Badge variant="secondary" className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    queued
                                  </Badge>
                                )}
                              </div>
                              <p className="text-sm text-gray-600 mb-1">{filing.company_name}</p>
                              <p className="text-xs text-gray-500">
                                Accession: {filing.accession_number}
                                {filing.file_size && (
                                  <span className="ml-2">
                                    Size: {(filing.file_size / 1024 / 1024).toFixed(1)} MB
                                  </span>
                                )}
                              </p>
                              
                              {/* Individual Document Progress */}
                              {isBeingProcessed && documentProgress && (
                                <div className="mt-3 space-y-2">
                                  <div className="flex justify-between items-center text-sm">
                                    <span>Progress:</span>
                                    <span>{Math.round(documentProgress.progress_percent)}%</span>
                                  </div>
                                  <Progress value={documentProgress.progress_percent} className="w-full h-2" />
                                  
                                  {/* Additional Progress Details */}
                                  {(documentProgress.chunks_created > 0 || documentProgress.tokens_used > 0) && (
                                    <div className="text-xs text-gray-500 flex gap-4">
                                      {documentProgress.chunks_created > 0 && (
                                        <span>Chunks: {documentProgress.chunks_created}</span>
                                      )}
                                      {documentProgress.tokens_used > 0 && (
                                        <span>Tokens: {documentProgress.tokens_used.toLocaleString()}</span>
                                      )}
                                    </div>
                                  )}
                                  
                                  {/* Error Message */}
                                  {documentProgress.error_message && (
                                    <div className="text-xs text-red-600 flex items-center gap-1 mt-1">
                                      <AlertCircle className="h-3 w-3" />
                                      {documentProgress.error_message}
                                    </div>
                                  )}
                                </div>
                              )}
                              
                              {/* Show queued status for selected but not yet processing */}
                              {isBeingProcessed && !documentProgress && (
                                <div className="mt-3">
                                  <div className="text-xs text-gray-500 flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    Waiting to start processing...
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          
                          {filing.document_url && (
                            <Button variant="outline" size="sm" asChild>
                              <a href={filing.document_url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="h-4 w-4 mr-1" />
                                View
                              </a>
                            </Button>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Results Message */}
      {!loading && filings.length === 0 && searchFilters.ticker && (
        <Card>
          <CardContent className="p-6 text-center">
            <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <p className="text-gray-500">
              No filings found for the specified criteria. Try different search parameters.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SECDocumentSearch;
