import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Layers, 
  BarChart3, 
  MapPin, 
  Hash,
  ArrowLeft,
  Calendar,
  Building,
  RefreshCw,
  Eye
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ChunkData {
  chunk_id: string;
  content: string;
  content_length: number;
  page_number: number | null;
  section_type: string;
  credibility_score: number;
  citation_info: string;
  search_score: number;
}

interface DocumentInfo {
  document_id: string;
  company: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  total_chunks: number;
  cik: string;
  processed_at: string;
}

interface ChunkStats {
  total_chunks: number;
  avg_chunk_length: number;
  total_content_length: number;
  page_range: {
    min: number | null;
    max: number | null;
  };
  section_types: string[];
  avg_credibility_score: number;
}

interface ChunkVisualizationData {
  document_id: string;
  document_info: DocumentInfo;
  chunks: ChunkData[];
  chunk_stats: ChunkStats;
}

interface SECChunkVisualizationProps {
  documentId: string;
  onBack?: () => void;
}

const SECChunkVisualization: React.FC<SECChunkVisualizationProps> = ({ documentId, onBack }) => {
  const [data, setData] = useState<ChunkVisualizationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState<ChunkData | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const { toast } = useToast();

  useEffect(() => {
    if (documentId) {
      loadChunkData();
    }
  }, [documentId]);

  const loadChunkData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/sec/documents/${documentId}/chunks`);
      if (!response.ok) {
        throw new Error('Failed to load chunk data');
      }

      const chunkData: ChunkVisualizationData = await response.json();
      setData(chunkData);

    } catch (error) {
      console.error('Error loading chunk data:', error);
      toast({
        title: "Error",
        description: "Failed to load chunk visualization data",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString();
  };

  const getChunksByPage = () => {
    if (!data) return {};
    
    const chunksByPage: { [key: number]: ChunkData[] } = {};
    data.chunks.forEach(chunk => {
      const page = chunk.page_number || 0;
      if (!chunksByPage[page]) {
        chunksByPage[page] = [];
      }
      chunksByPage[page].push(chunk);
    });
    
    return chunksByPage;
  };

  const getChunksBySection = () => {
    if (!data) return {};
    
    const chunksBySection: { [key: string]: ChunkData[] } = {};
    data.chunks.forEach(chunk => {
      const section = chunk.section_type || 'Unknown';
      if (!chunksBySection[section]) {
        chunksBySection[section] = [];
      }
      chunksBySection[section].push(chunk);
    });
    
    return chunksBySection;
  };
  const getCredibilityVariant = (score: number): "default" | "secondary" | "destructive" | "outline" => {
    if (score >= 0.8) return 'default';
    if (score >= 0.6) return 'secondary';
    return 'destructive';
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center p-8">
          <RefreshCw className="h-8 w-8 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <div className="text-muted-foreground">No chunk data available</div>
          <Button onClick={loadChunkData} className="mt-4">
            Load Data
          </Button>
        </CardContent>
      </Card>
    );
  }

  const chunksByPage = getChunksByPage();
  const chunksBySection = getChunksBySection();

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {onBack && (
                <Button variant="outline" size="sm" onClick={onBack}>
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              )}
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Layers className="h-5 w-5" />
                  Chunk Visualization
                </CardTitle>
                <CardDescription>
                  Document analysis and chunk breakdown
                </CardDescription>
              </div>
            </div>
            <Button onClick={loadChunkData} disabled={loading} variant="outline" size="sm">
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold text-lg">{data.document_info.company}</h3>
              <p className="text-muted-foreground">
                {data.document_info.ticker} • {data.document_info.form_type}
              </p>
              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2">
                <span className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  {formatDate(data.document_info.filing_date)}
                </span>
                <span className="flex items-center gap-1">
                  <Building className="h-4 w-4" />
                  CIK: {data.document_info.cik}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{data.chunk_stats.total_chunks}</div>
                <div className="text-sm text-muted-foreground">Total Chunks</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{data.chunk_stats.avg_chunk_length}</div>
                <div className="text-sm text-muted-foreground">Avg Length</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="chunks">Chunks</TabsTrigger>
          <TabsTrigger value="pages">By Page</TabsTrigger>
          <TabsTrigger value="sections">By Section</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Statistics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Document Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span>Total Content Length</span>
                  <span className="font-mono">{data.chunk_stats.total_content_length.toLocaleString()} chars</span>
                </div>
                <div className="flex justify-between">
                  <span>Average Chunk Length</span>
                  <span className="font-mono">{data.chunk_stats.avg_chunk_length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Page Range</span>
                  <span className="font-mono">
                    {data.chunk_stats.page_range.min || 'N/A'} - {data.chunk_stats.page_range.max || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Average Credibility</span>
                  <Badge variant={getCredibilityVariant(data.chunk_stats.avg_credibility_score)}>
                    {(data.chunk_stats.avg_credibility_score * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Section Types</span>
                  <span>{data.chunk_stats.section_types.length}</span>
                </div>
              </CardContent>
            </Card>

            {/* Section Types */}
            <Card>
              <CardHeader>
                <CardTitle>Section Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(chunksBySection).map(([section, chunks]) => (
                    <div key={section} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">{section || 'Unknown'}</span>
                        <span className="text-muted-foreground">{chunks.length} chunks</span>
                      </div>
                      <Progress 
                        value={(chunks.length / data.chunk_stats.total_chunks) * 100} 
                        className="h-2" 
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Chunks Tab */}
        <TabsContent value="chunks">
          <Card>
            <CardHeader>
              <CardTitle>All Chunks ({data.chunks.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {data.chunks.map((chunk, index) => (
                    <Card 
                      key={chunk.chunk_id}
                      className={`cursor-pointer transition-colors ${
                        selectedChunk?.chunk_id === chunk.chunk_id ? 'ring-2 ring-blue-500' : ''
                      }`}
                      onClick={() => setSelectedChunk(chunk)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="outline">#{index + 1}</Badge>
                              {chunk.page_number && (
                                <Badge variant="secondary">Page {chunk.page_number}</Badge>
                              )}
                              {chunk.section_type && (
                                <Badge variant="outline">{chunk.section_type}</Badge>
                              )}
                              <Badge variant={getCredibilityVariant(chunk.credibility_score)}>
                                {(chunk.credibility_score * 100).toFixed(0)}%
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground truncate">
                              {chunk.content}
                            </p>
                            <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                              <span>{chunk.content_length} chars</span>
                              {chunk.citation_info && (
                                <span className="truncate">{chunk.citation_info}</span>
                              )}
                            </div>
                          </div>
                          <Button variant="ghost" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* By Page Tab */}
        <TabsContent value="pages">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5" />
                Chunks by Page
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {Object.entries(chunksByPage)
                    .sort(([a], [b]) => Number(a) - Number(b))
                    .map(([page, chunks]) => (
                    <Card key={page}>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                          Page {page === '0' ? 'Unknown' : page}
                          <Badge variant="secondary" className="ml-2">
                            {chunks.length} chunks
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {chunks.map((chunk, index) => (
                            <div key={chunk.chunk_id} className="text-sm p-2 bg-muted rounded">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-mono text-xs">#{index + 1}</span>
                                <Badge variant={getCredibilityVariant(chunk.credibility_score)} className="text-xs">
                                  {(chunk.credibility_score * 100).toFixed(0)}%
                                </Badge>
                              </div>
                              <p className="text-muted-foreground">
                                {chunk.content.length > 100 
                                  ? `${chunk.content.substring(0, 100)}...` 
                                  : chunk.content
                                }
                              </p>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* By Section Tab */}
        <TabsContent value="sections">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Hash className="h-5 w-5" />
                Chunks by Section
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {Object.entries(chunksBySection).map(([section, chunks]) => (
                    <Card key={section}>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">
                          {section || 'Unknown Section'}
                          <Badge variant="secondary" className="ml-2">
                            {chunks.length} chunks
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {chunks.map((chunk, index) => (
                            <div key={chunk.chunk_id} className="text-sm p-2 bg-muted rounded">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-mono text-xs">
                                  #{index + 1} {chunk.page_number ? `(Page ${chunk.page_number})` : ''}
                                </span>
                                <Badge variant={getCredibilityVariant(chunk.credibility_score)} className="text-xs">
                                  {(chunk.credibility_score * 100).toFixed(0)}%
                                </Badge>
                              </div>
                              <p className="text-muted-foreground">
                                {chunk.content.length > 100 
                                  ? `${chunk.content.substring(0, 100)}...` 
                                  : chunk.content
                                }
                              </p>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Selected Chunk Detail */}
      {selectedChunk && (
        <Card>
          <CardHeader>
            <CardTitle>Chunk Details</CardTitle>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setSelectedChunk(null)}
              className="w-fit"
            >
              Close
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant="outline">{selectedChunk.chunk_id}</Badge>
                {selectedChunk.page_number && (
                  <Badge variant="secondary">Page {selectedChunk.page_number}</Badge>
                )}
                {selectedChunk.section_type && (
                  <Badge variant="outline">{selectedChunk.section_type}</Badge>
                )}
                <Badge variant={getCredibilityVariant(selectedChunk.credibility_score)}>
                  Credibility: {(selectedChunk.credibility_score * 100).toFixed(1)}%
                </Badge>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Content</h4>
                <div className="bg-muted p-4 rounded-lg">
                  <p className="text-sm">{selectedChunk.content}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Length:</span> {selectedChunk.content_length} characters
                </div>
                <div>
                  <span className="font-medium">Search Score:</span> {selectedChunk.search_score.toFixed(3)}
                </div>
              </div>
              {selectedChunk.citation_info && (
                <div>
                  <span className="font-medium">Citation:</span> {selectedChunk.citation_info}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SECChunkVisualization;
