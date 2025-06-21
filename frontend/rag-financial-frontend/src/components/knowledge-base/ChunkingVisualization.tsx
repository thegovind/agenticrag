import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  FileText, 
  Layers, 
  BarChart3, 
  Table, 
  Hash, 
  Eye, 
  Download,
  ChevronRight,
  ChevronDown,
  MapPin,
  Zap
} from 'lucide-react';

interface ChunkMetadata {
  id: string;
  content: string;
  startPage: number;
  endPage: number;
  section: string;
  subsection?: string;
  chunkType: 'text' | 'table' | 'chart' | 'footnote' | 'header';
  size: number;
  overlap: number;
  confidence: number;
  citations: string[];
}

interface DocumentStructure {
  id: string;
  filename: string;
  totalPages: number;
  sections: {
    name: string;
    startPage: number;
    endPage: number;
    subsections: {
      name: string;
      startPage: number;
      endPage: number;
      chunks: ChunkMetadata[];
    }[];
  }[];
  processingStatus: 'processing' | 'completed' | 'failed';
  processingProgress: number;
}

const ChunkingVisualization: React.FC = () => {
  const [selectedDocument, setSelectedDocument] = useState<string>('');
  const [documentStructure, setDocumentStructure] = useState<DocumentStructure | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<ChunkMetadata | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [chunkingStrategy, setChunkingStrategy] = useState<'hierarchical' | 'semantic' | 'hybrid'>('hierarchical');

  const mockDocuments = [
    { id: '1', name: 'AAPL_10K_2023.pdf' },
    { id: '2', name: 'MSFT_10Q_Q3_2023.pdf' },
    { id: '3', name: 'GOOGL_Annual_Report_2023.pdf' }
  ];

  const mockDocumentStructure: DocumentStructure = {
    id: '1',
    filename: 'AAPL_10K_2023.pdf',
    totalPages: 112,
    processingStatus: 'completed',
    processingProgress: 100,
    sections: [
      {
        name: 'Business Overview',
        startPage: 1,
        endPage: 15,
        subsections: [
          {
            name: 'Company Description',
            startPage: 1,
            endPage: 5,
            chunks: [
              {
                id: 'chunk_1',
                content: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide...',
                startPage: 1,
                endPage: 2,
                section: 'Business Overview',
                subsection: 'Company Description',
                chunkType: 'text',
                size: 512,
                overlap: 50,
                confidence: 0.95,
                citations: ['Page 1-2, Business Overview']
              },
              {
                id: 'chunk_2',
                content: 'The Company serves consumers and businesses worldwide through its retail and online stores, direct sales force...',
                startPage: 2,
                endPage: 3,
                section: 'Business Overview',
                subsection: 'Company Description',
                chunkType: 'text',
                size: 487,
                overlap: 50,
                confidence: 0.92,
                citations: ['Page 2-3, Business Overview']
              }
            ]
          },
          {
            name: 'Products and Services',
            startPage: 6,
            endPage: 15,
            chunks: [
              {
                id: 'chunk_3',
                content: 'iPhone revenue for fiscal 2023 was $200.6 billion, representing 52% of total net sales...',
                startPage: 8,
                endPage: 8,
                section: 'Business Overview',
                subsection: 'Products and Services',
                chunkType: 'text',
                size: 445,
                overlap: 50,
                confidence: 0.98,
                citations: ['Page 8, Products and Services']
              }
            ]
          }
        ]
      },
      {
        name: 'Financial Information',
        startPage: 16,
        endPage: 45,
        subsections: [
          {
            name: 'Consolidated Statements',
            startPage: 16,
            endPage: 25,
            chunks: [
              {
                id: 'chunk_4',
                content: 'Revenue breakdown by product category and geographic region for fiscal year 2023...',
                startPage: 18,
                endPage: 19,
                section: 'Financial Information',
                subsection: 'Consolidated Statements',
                chunkType: 'table',
                size: 678,
                overlap: 25,
                confidence: 0.89,
                citations: ['Page 18-19, Table 1: Revenue by Product Category']
              }
            ]
          }
        ]
      }
    ]
  };

  useEffect(() => {
    if (selectedDocument) {
      setDocumentStructure(mockDocumentStructure);
    }
  }, [selectedDocument]);

  const toggleSection = (sectionName: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionName)) {
      newExpanded.delete(sectionName);
    } else {
      newExpanded.add(sectionName);
    }
    setExpandedSections(newExpanded);
  };

  const getChunkTypeIcon = (type: ChunkMetadata['chunkType']) => {
    switch (type) {
      case 'text': return <FileText className="h-4 w-4" />;
      case 'table': return <Table className="h-4 w-4" />;
      case 'chart': return <BarChart3 className="h-4 w-4" />;
      case 'footnote': return <Hash className="h-4 w-4" />;
      case 'header': return <Layers className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getChunkTypeColor = (type: ChunkMetadata['chunkType']) => {
    switch (type) {
      case 'text': return 'bg-blue-100 text-blue-800';
      case 'table': return 'bg-green-100 text-green-800';
      case 'chart': return 'bg-purple-100 text-purple-800';
      case 'footnote': return 'bg-yellow-100 text-yellow-800';
      case 'header': return 'bg-gray-100 text-gray-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Document Chunking Visualization</h2>
          <p className="text-muted-foreground">
            Visualize how financial documents are processed and chunked for RAG analysis
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {chunkingStrategy.charAt(0).toUpperCase() + chunkingStrategy.slice(1)} Strategy
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Document Selection
            </CardTitle>
            <CardDescription>
              Select a document to visualize its chunking structure
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              {mockDocuments.map((doc) => (
                <Button
                  key={doc.id}
                  variant={selectedDocument === doc.id ? "default" : "outline"}
                  className="w-full justify-start"
                  onClick={() => setSelectedDocument(doc.id)}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  {doc.name}
                </Button>
              ))}
            </div>

            {documentStructure && (
              <div className="space-y-3 pt-4 border-t">
                <div className="flex items-center justify-between text-sm">
                  <span>Processing Status:</span>
                  <Badge variant={documentStructure.processingStatus === 'completed' ? 'default' : 'secondary'}>
                    {documentStructure.processingStatus}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Progress:</span>
                    <span>{documentStructure.processingProgress}%</span>
                  </div>
                  <Progress value={documentStructure.processingProgress} className="h-2" />
                </div>
                <div className="text-sm text-muted-foreground">
                  {documentStructure.totalPages} pages • {documentStructure.sections.length} sections
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Document Structure
            </CardTitle>
            <CardDescription>
              Hierarchical view of document sections and chunks
            </CardDescription>
          </CardHeader>
          <CardContent>
            {documentStructure ? (
              <ScrollArea className="h-[600px] pr-4">
                <div className="space-y-4">
                  {documentStructure.sections.map((section) => (
                    <div key={section.name} className="border rounded-lg p-4">
                      <div
                        className="flex items-center justify-between cursor-pointer"
                        onClick={() => toggleSection(section.name)}
                      >
                        <div className="flex items-center gap-2">
                          {expandedSections.has(section.name) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                          <h3 className="font-semibold">{section.name}</h3>
                          <Badge variant="outline" className="text-xs">
                            Pages {section.startPage}-{section.endPage}
                          </Badge>
                        </div>
                        <Badge variant="secondary">
                          {section.subsections.reduce((acc, sub) => acc + sub.chunks.length, 0)} chunks
                        </Badge>
                      </div>

                      {expandedSections.has(section.name) && (
                        <div className="mt-4 space-y-3">
                          {section.subsections.map((subsection) => (
                            <div key={subsection.name} className="ml-6 border-l-2 border-gray-200 pl-4">
                              <div className="flex items-center gap-2 mb-2">
                                <h4 className="font-medium text-sm">{subsection.name}</h4>
                                <Badge variant="outline" className="text-xs">
                                  Pages {subsection.startPage}-{subsection.endPage}
                                </Badge>
                              </div>
                              <div className="space-y-2">
                                {subsection.chunks.map((chunk) => (
                                  <div
                                    key={chunk.id}
                                    className={`p-3 rounded border cursor-pointer transition-colors ${
                                      selectedChunk?.id === chunk.id
                                        ? 'border-blue-500 bg-blue-50'
                                        : 'border-gray-200 hover:border-gray-300'
                                    }`}
                                    onClick={() => setSelectedChunk(chunk)}
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <div className="flex items-center gap-2">
                                        {getChunkTypeIcon(chunk.chunkType)}
                                        <Badge className={`text-xs ${getChunkTypeColor(chunk.chunkType)}`}>
                                          {chunk.chunkType}
                                        </Badge>
                                        <span className="text-xs text-muted-foreground">
                                          {chunk.size} chars
                                        </span>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <span className={`text-xs font-medium ${getConfidenceColor(chunk.confidence)}`}>
                                          {(chunk.confidence * 100).toFixed(0)}%
                                        </span>
                                        <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                                          <Eye className="h-3 w-3" />
                                        </Button>
                                      </div>
                                    </div>
                                    <p className="text-xs text-muted-foreground line-clamp-2">
                                      {chunk.content}
                                    </p>
                                    <div className="flex items-center gap-2 mt-2">
                                      <MapPin className="h-3 w-3 text-muted-foreground" />
                                      <span className="text-xs text-muted-foreground">
                                        Pages {chunk.startPage}-{chunk.endPage}
                                      </span>
                                      {chunk.overlap > 0 && (
                                        <Badge variant="outline" className="text-xs">
                                          {chunk.overlap} char overlap
                                        </Badge>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="flex items-center justify-center h-[400px] text-muted-foreground">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Select a document to view its chunking structure</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {selectedChunk && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Chunk Details
            </CardTitle>
            <CardDescription>
              Detailed information about the selected chunk
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="content" className="w-full">
              <TabsList>
                <TabsTrigger value="content">Content</TabsTrigger>
                <TabsTrigger value="metadata">Metadata</TabsTrigger>
                <TabsTrigger value="citations">Citations</TabsTrigger>
              </TabsList>
              
              <TabsContent value="content" className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  {getChunkTypeIcon(selectedChunk.chunkType)}
                  <Badge className={getChunkTypeColor(selectedChunk.chunkType)}>
                    {selectedChunk.chunkType}
                  </Badge>
                  <Badge variant="outline">
                    {selectedChunk.size} characters
                  </Badge>
                  <Badge variant="outline" className={getConfidenceColor(selectedChunk.confidence)}>
                    {(selectedChunk.confidence * 100).toFixed(0)}% confidence
                  </Badge>
                </div>
                <ScrollArea className="h-[300px] w-full border rounded p-4">
                  <p className="text-sm leading-relaxed">{selectedChunk.content}</p>
                </ScrollArea>
              </TabsContent>
              
              <TabsContent value="metadata" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chunk ID</label>
                    <p className="text-sm text-muted-foreground">{selectedChunk.id}</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Section</label>
                    <p className="text-sm text-muted-foreground">{selectedChunk.section}</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Subsection</label>
                    <p className="text-sm text-muted-foreground">{selectedChunk.subsection || 'N/A'}</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Page Range</label>
                    <p className="text-sm text-muted-foreground">
                      {selectedChunk.startPage}-{selectedChunk.endPage}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chunk Size</label>
                    <p className="text-sm text-muted-foreground">{selectedChunk.size} characters</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Overlap</label>
                    <p className="text-sm text-muted-foreground">{selectedChunk.overlap} characters</p>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="citations" className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Citation References</label>
                  <div className="space-y-2">
                    {selectedChunk.citations.map((citation, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 border rounded">
                        <MapPin className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{citation}</span>
                        <Button size="sm" variant="ghost" className="ml-auto">
                          <Download className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ChunkingVisualization;
