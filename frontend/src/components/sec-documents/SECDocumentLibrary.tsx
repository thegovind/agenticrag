import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { FileText, Calendar, Eye, RefreshCw, Search, Filter } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SECDocument {
  document_id: string;
  company: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  chunk_count: number;
  processed_at: string;
  source: string;
  cik: string;
}

interface SECDocumentLibraryData {
  documents: SECDocument[];
  total_count: number;
  total_chunks: number;
  companies: string[];
  form_types: string[];
}

interface SECDocumentLibraryProps {
  onViewChunks?: (documentId: string) => void;
}

const SECDocumentLibrary: React.FC<SECDocumentLibraryProps> = ({ onViewChunks }) => {
  const [documents, setDocuments] = useState<SECDocument[]>([]);
  const [filteredDocuments, setFilteredDocuments] = useState<SECDocument[]>([]);
  const [loading, setLoading] = useState(false);  const [selectedCompany, setSelectedCompany] = useState<string>('all');
  const [selectedFormType, setSelectedFormType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [availableCompanies, setAvailableCompanies] = useState<string[]>([]);
  const [availableFormTypes, setAvailableFormTypes] = useState<string[]>([]);
  const [totalStats, setTotalStats] = useState({ count: 0, chunks: 0 });
  const { toast } = useToast();
  useEffect(() => {
    console.log('Document Library: Component mounted, loading library');
    loadDocumentLibrary();
  }, []);

  useEffect(() => {
    console.log('Document Library: Applying filters, documents:', documents.length);
    applyFilters();
  }, [documents, selectedCompany, selectedFormType, searchTerm]);  const loadDocumentLibrary = async () => {
    console.log('Document Library: Starting to load library');
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (selectedCompany && selectedCompany !== 'all') params.append('company', selectedCompany);
      if (selectedFormType && selectedFormType !== 'all') params.append('form_type', selectedFormType);
      params.append('limit', '1000');

      console.log('Loading document library with params:', params.toString());
      const response = await fetch(`/api/v1/sec/library?${params.toString()}`);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Library API error:', response.status, errorText);
        throw new Error(`Failed to load document library: ${response.status}`);
      }

      const data: SECDocumentLibraryData = await response.json();
      console.log('Received library data:', data);
      
      setDocuments(data.documents || []);
      setAvailableCompanies(data.companies || []);
      setAvailableFormTypes(data.form_types || []);
      setTotalStats({ count: data.total_count || 0, chunks: data.total_chunks || 0 });

      toast({
        title: "Library Loaded",
        description: `Loaded ${data.total_count || 0} documents with ${data.total_chunks || 0} chunks`,
      });

    } catch (error) {
      console.error('Error loading document library:', error);
      toast({
        title: "Error",
        description: `Failed to load document library: ${error}`,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
      console.log('Document Library: Loading finished');
    }
  };  const applyFilters = () => {
    console.log('Document Library: Applying filters', { 
      documents: documents.length, 
      selectedCompany, 
      selectedFormType, 
      searchTerm 
    });
    
    let filtered = documents;

    if (selectedCompany && selectedCompany !== 'all') {
      filtered = filtered.filter(doc => doc.company && doc.company === selectedCompany);
      console.log('Document Library: After company filter:', filtered.length);
    }

    if (selectedFormType && selectedFormType !== 'all') {
      filtered = filtered.filter(doc => doc.form_type && doc.form_type === selectedFormType);
      console.log('Document Library: After form type filter:', filtered.length);
    }

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(doc => 
        (doc.company && doc.company.toLowerCase().includes(term)) ||
        (doc.ticker && doc.ticker.toLowerCase().includes(term)) ||
        (doc.form_type && doc.form_type.toLowerCase().includes(term)) ||
        (doc.accession_number && doc.accession_number.toLowerCase().includes(term)) ||
        (doc.document_id && doc.document_id.toLowerCase().includes(term))
      );
      console.log('Document Library: After search filter:', filtered.length);
    }

    console.log('Document Library: Final filtered documents:', filtered.length);
    setFilteredDocuments(filtered);
  };
  const clearFilters = () => {
    setSelectedCompany('all');
    setSelectedFormType('all');
    setSearchTerm('');
  };
  const formatDate = (dateStr: string) => {
    try {
      return dateStr ? new Date(dateStr).toLocaleDateString() : 'N/A';
    } catch {
      return 'Invalid Date';
    }
  };

  const getStatusBadge = (chunkCount: number) => {
    if (chunkCount > 100) {
      return <Badge variant="default">Large ({chunkCount} chunks)</Badge>;
    } else if (chunkCount > 50) {
      return <Badge variant="secondary">Medium ({chunkCount} chunks)</Badge>;
    } else {
      return <Badge variant="outline">Small ({chunkCount} chunks)</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            SEC Document Library
          </CardTitle>
          <CardDescription>
            Browse and analyze SEC documents in the vector store
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{totalStats.count}</div>
              <div className="text-sm text-muted-foreground">Total Documents</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{totalStats.chunks}</div>
              <div className="text-sm text-muted-foreground">Total Chunks</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{availableCompanies.length}</div>
              <div className="text-sm text-muted-foreground">Companies</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{availableFormTypes.length}</div>
              <div className="text-sm text-muted-foreground">Form Types</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by company, ticker, form type, or accession number..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>            <Select value={selectedCompany} onValueChange={setSelectedCompany}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by Company" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Companies</SelectItem>
                {availableCompanies.map(company => (
                  <SelectItem key={company} value={company}>{company}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedFormType} onValueChange={setSelectedFormType}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Form Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Forms</SelectItem>
                {availableFormTypes.map(formType => (
                  <SelectItem key={formType} value={formType}>{formType}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={clearFilters} className="gap-2">
              <Filter className="h-4 w-4" />
              Clear
            </Button>
            <Button onClick={loadDocumentLibrary} disabled={loading} className="gap-2">
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Documents Table */}
      <Card>
        <CardHeader>
          <CardTitle>Documents ({filteredDocuments.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Company</TableHead>
                  <TableHead>Form Type</TableHead>
                  <TableHead>Filing Date</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead>Processed</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredDocuments.map((document) => (                  <TableRow key={document.document_id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{document.company || 'Unknown Company'}</div>
                        <div className="text-sm text-muted-foreground">
                          {document.ticker || 'N/A'} • CIK: {document.cik || 'N/A'}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{document.form_type || 'N/A'}</Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-muted-foreground" />
                        {document.filing_date ? formatDate(document.filing_date) : 'N/A'}
                      </div>
                    </TableCell>
                    <TableCell>
                      {getStatusBadge(document.chunk_count || 0)}
                    </TableCell>
                    <TableCell>
                      <div className="text-sm text-muted-foreground">
                        {document.processed_at ? formatDate(document.processed_at) : 'N/A'}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onViewChunks?.(document.document_id)}
                        className="gap-2"
                      >
                        <Eye className="h-4 w-4" />
                        View Chunks
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          
          {filteredDocuments.length === 0 && !loading && (
            <div className="text-center py-8 text-muted-foreground">
              No documents found matching the current filters
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SECDocumentLibrary;
