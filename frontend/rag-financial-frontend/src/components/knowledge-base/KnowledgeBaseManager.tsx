import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Upload, FileText, CheckCircle, AlertCircle, Clock, Trash2, Eye, Download } from 'lucide-react';
import ChunkingVisualization from './ChunkingVisualization';
import { ModelSettings } from '../shared/ModelConfiguration';

interface Document {
  id: string;
  filename: string;
  type: string;
  size: number;
  uploadDate: string;
  status: 'processing' | 'completed' | 'failed' | 'pending';
  chunks: number;
  conflicts?: number;
  processingProgress?: number;
}

interface ConflictItem {
  id: string;
  documentId: string;
  chunkId: string;
  conflictType: 'duplicate' | 'contradiction' | 'outdated';
  description: string;
  sources: string[];
  status: 'pending' | 'resolved' | 'ignored';
}

interface KnowledgeBaseManagerProps {
  modelSettings: ModelSettings;
}

const KnowledgeBaseManager: React.FC<KnowledgeBaseManagerProps> = ({ modelSettings }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [conflicts, setConflicts] = useState<ConflictItem[]>([]);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    fetchDocuments();
    fetchConflicts();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      setDocuments([
        {
          id: '1',
          filename: 'AAPL_10K_2023.pdf',
          type: '10-K',
          size: 2048576,
          uploadDate: '2024-01-15T10:30:00Z',
          status: 'completed',
          chunks: 156,
          conflicts: 2
        },
        {
          id: '2',
          filename: 'MSFT_10Q_Q3_2023.pdf',
          type: '10-Q',
          size: 1536000,
          uploadDate: '2024-01-14T14:20:00Z',
          status: 'processing',
          chunks: 89,
          processingProgress: 75
        },
        {
          id: '3',
          filename: 'GOOGL_Annual_Report_2023.pdf',
          type: 'Annual Report',
          size: 3072000,
          uploadDate: '2024-01-13T09:15:00Z',
          status: 'failed',
          chunks: 0,
          conflicts: 0
        }
      ]);
    }
  };

  const fetchConflicts = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/knowledge-base/conflicts`);
      if (response.ok) {
        const data = await response.json();
        setConflicts(data.conflicts || []);
      }
    } catch (error) {
      console.error('Failed to fetch conflicts:', error);
      setConflicts([
        {
          id: '1',
          documentId: '1',
          chunkId: 'chunk_156',
          conflictType: 'contradiction',
          description: 'Revenue figures differ between Q3 and annual report',
          sources: ['AAPL_10K_2023.pdf', 'AAPL_10Q_Q3_2023.pdf'],
          status: 'pending'
        },
        {
          id: '2',
          documentId: '1',
          chunkId: 'chunk_89',
          conflictType: 'duplicate',
          description: 'Similar content found in multiple documents',
          sources: ['AAPL_10K_2023.pdf', 'MSFT_10K_2023.pdf'],
          status: 'pending'
        }
      ]);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    Array.from(selectedFiles).forEach(file => {
      formData.append('files', file);
    });

    try {
      const apiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Upload successful:', data);
        fetchDocuments();
        setSelectedFiles(null);
      } else {
        console.error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsUploading(false);
            fetchDocuments();
            return 100;
          }
          return prev + 10;
        });
      }, 200);
    }

    setIsUploading(false);
  };

  const handleDeleteDocument = async (documentId: string) => {
    try {
      const apiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        fetchDocuments();
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const handleResolveConflict = async (conflictId: string, resolution: 'resolve' | 'ignore') => {
    try {
      const apiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL || '/api/v1';
      const response = await fetch(`${apiBaseUrl}/knowledge-base/conflicts/${conflictId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: resolution === 'resolve' ? 'resolved' : 'ignored' }),
      });

      if (response.ok) {
        fetchConflicts();
      }
    } catch (error) {
      console.error('Conflict resolution failed:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      completed: 'default',
      processing: 'secondary',
      failed: 'destructive',
      pending: 'outline'
    } as const;

    return (
      <Badge variant={variants[status as keyof typeof variants] || 'outline'}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Knowledge Base Management</h1>
          <p className="text-muted-foreground">
            Upload, process, and manage financial documents for RAG analysis
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="upload">Document Upload</TabsTrigger>
          <TabsTrigger value="documents">Document Library</TabsTrigger>
          <TabsTrigger value="chunking">Chunking Visualization</TabsTrigger>
          <TabsTrigger value="conflicts">Conflict Resolution</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Financial Documents
              </CardTitle>
              <CardDescription>
                Upload 10-K, 10-Q, and other financial reports for processing and analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="documents">Select Documents</Label>
                <Input
                  id="documents"
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt"
                  onChange={(e) => setSelectedFiles(e.target.files)}
                  disabled={isUploading}
                />
                <p className="text-sm text-muted-foreground">
                  Supported formats: PDF, DOCX, TXT (Max 10MB per file)
                </p>
              </div>

              {selectedFiles && selectedFiles.length > 0 && (
                <div className="space-y-2">
                  <Label>Selected Files:</Label>
                  {Array.from(selectedFiles).map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        <span className="text-sm">{file.name}</span>
                        <span className="text-xs text-muted-foreground">
                          ({formatFileSize(file.size)})
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {isUploading && (
                <div className="space-y-2">
                  <Label>Upload Progress</Label>
                  <Progress value={uploadProgress} className="w-full" />
                  <p className="text-sm text-muted-foreground">{uploadProgress}% complete</p>
                </div>
              )}

              <Button
                onClick={handleFileUpload}
                disabled={!selectedFiles || selectedFiles.length === 0 || isUploading}
                className="w-full"
              >
                {isUploading ? 'Uploading...' : 'Upload Documents'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="documents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Document Library</CardTitle>
              <CardDescription>
                Manage uploaded documents and their processing status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Upload Date</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Chunks</TableHead>
                    <TableHead>Conflicts</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {documents.map((doc) => (
                    <TableRow key={doc.id}>
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          {doc.filename}
                        </div>
                      </TableCell>
                      <TableCell>{doc.type}</TableCell>
                      <TableCell>{formatFileSize(doc.size)}</TableCell>
                      <TableCell>{formatDate(doc.uploadDate)}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(doc.status)}
                          {getStatusBadge(doc.status)}
                          {doc.status === 'processing' && doc.processingProgress && (
                            <span className="text-xs text-muted-foreground">
                              ({doc.processingProgress}%)
                            </span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>{doc.chunks}</TableCell>
                      <TableCell>
                        {doc.conflicts ? (
                          <Badge variant="destructive">{doc.conflicts}</Badge>
                        ) : (
                          <Badge variant="outline">0</Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button variant="outline" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <Download className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDeleteDocument(doc.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chunking" className="space-y-4">
          <ChunkingVisualization />
        </TabsContent>

        <TabsContent value="conflicts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Conflict Resolution</CardTitle>
              <CardDescription>
                Review and resolve conflicts between documents and data sources
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {conflicts.map((conflict) => (
                  <Card key={conflict.id}>
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Badge variant={conflict.conflictType === 'contradiction' ? 'destructive' : 'secondary'}>
                              {conflict.conflictType}
                            </Badge>
                            {getStatusBadge(conflict.status)}
                          </div>
                          <p className="text-sm font-medium">{conflict.description}</p>
                          <div className="text-xs text-muted-foreground">
                            <p>Sources: {conflict.sources.join(', ')}</p>
                            <p>Chunk ID: {conflict.chunkId}</p>
                          </div>
                        </div>
                        {conflict.status === 'pending' && (
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={() => handleResolveConflict(conflict.id, 'resolve')}
                            >
                              Resolve
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleResolveConflict(conflict.id, 'ignore')}
                            >
                              Ignore
                            </Button>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
                <FileText className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{documents.length}</div>
                <p className="text-xs text-muted-foreground">
                  +2 from last week
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Chunks</CardTitle>
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {documents.reduce((sum, doc) => sum + doc.chunks, 0)}
                </div>
                <p className="text-xs text-muted-foreground">
                  Processed chunks
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Conflicts</CardTitle>
                <AlertCircle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {conflicts.filter(c => c.status === 'pending').length}
                </div>
                <p className="text-xs text-muted-foreground">
                  Require attention
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Processing Rate</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">94%</div>
                <p className="text-xs text-muted-foreground">
                  Success rate
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default KnowledgeBaseManager;
