import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Search, Download, FileText, Calendar, ExternalLink, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

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
  years: number[];  // Changed from single year to multiple years
}

const SECDocumentSearch: React.FC = () => {  // Search form state
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    ticker: '',
    formTypes: ['10-K'],
    years: [new Date().getFullYear()]  // Changed to array with current year
  });
  
  // Results and processing state
  const [filings, setFilings] = useState<Filing[]>([]);
  const [selectedFilings, setSelectedFilings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState<string[]>([]);
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
    }    if (searchFilters.formTypes.length === 0) {
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

    try {      // Use the specific filings endpoint
      const response = await fetch('/api/v1/sec/filings/specific', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: searchFilters.ticker.toUpperCase(),
          form_types: searchFilters.formTypes, // Send all selected form types
          years: searchFilters.years, // Send selected years
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

    setProcessing(selectedFilings);

    try {
      // Prepare batch request
      const filingsToProcess = selectedFilings.map(accessionNumber => {
        const filing = filings.find(f => f.accession_number === accessionNumber);
        return {
          ticker: filing?.ticker || '',
          accession_number: accessionNumber
        };
      }).filter(f => f.ticker); // Filter out any invalid filings

      const response = await fetch('/api/v1/sec/documents/process-multiple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filings: filingsToProcess
        })
      });

      if (!response.ok) {
        throw new Error('Failed to process selected filings');
      }

      const result = await response.json();
      
      let description = '';
      if (result.processed > 0 && result.skipped > 0) {
        description = `Processed ${result.processed} new filing(s) and skipped ${result.skipped} already indexed filing(s). Created ${result.total_chunks_created} new chunks.`;
      } else if (result.processed > 0) {
        description = `Successfully processed ${result.processed} filing(s). Created ${result.total_chunks_created} chunks for the knowledge base.`;
      } else if (result.skipped > 0) {
        description = `All ${result.skipped} selected filing(s) were already indexed in the knowledge base.`;
      }

      toast({
        title: "Processing Complete",
        description: description,
      });

      // Clear selections after successful processing
      setSelectedFilings([]);

    } catch (error) {
      console.error('Processing error:', error);
      toast({
        title: "Processing Error",
        description: "Failed to process selected filings. Please try again.",
        variant: "destructive",
      });
    } finally {
      setProcessing([]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            SEC Document Search
          </CardTitle>
          <CardDescription>
            Search for SEC filings by company, document type, and year
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
          </div>          {/* Years Selection */}
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
          </div>          {/* Search Button */}
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
      </Card>

      {/* Search Results */}
      {filings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results</CardTitle>
            <CardDescription>
              Found {filings.length} filing(s). Select the documents you want to process.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Select All/None */}
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
                
                <Button
                  onClick={processSelectedFilings}
                  disabled={selectedFilings.length === 0 || processing.length > 0}
                  variant="default"
                >
                  {processing.length > 0 ? (
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
              </div>

              {/* Filing List */}
              <div className="space-y-3">
                {filings.map((filing) => (
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
                ))}
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
