import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Search, Library, BarChart3, Layers, Info } from 'lucide-react';

import SECDocumentSearch from './SECDocumentSearch';
import SECDocumentLibrary from './SECDocumentLibrary';
import SECAnalytics from './SECAnalytics';
import SECChunkVisualization from './SECChunkVisualization';

const SECDocumentsManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState('search');
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>('');

  const handleViewChunks = (documentId: string) => {
    setSelectedDocumentId(documentId);
    setActiveTab('visualization');
  };

  const handleBackFromVisualization = () => {
    setSelectedDocumentId('');
    setActiveTab('library');
  };

  return (
    <div className="container mx-auto p-6">
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl flex items-center gap-2">
                SEC Documents Management
                <Badge variant="secondary">AI-Powered</Badge>
              </CardTitle>
              <p className="text-muted-foreground mt-2">
                Search, process, and analyze SEC documents in the vector store with intelligent chunking and embeddings
              </p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                Vector Store Integration
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="search" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Search & Process
          </TabsTrigger>
          <TabsTrigger value="library" className="flex items-center gap-2">
            <Library className="h-4 w-4" />
            Document Library
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Chunk Visualization
          </TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-6">
          <Card className="mb-4">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                Search for SEC filings and process them into the vector store with parallel processing and real-time progress tracking.
              </div>
            </CardContent>
          </Card>
          <SECDocumentSearch />
        </TabsContent>        <TabsContent value="library" className="mt-6">
          <Card className="mb-4">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                Browse all SEC documents that have been processed and indexed in the vector store.
              </div>
            </CardContent>
          </Card>
          <SECDocumentLibrary onViewChunks={handleViewChunks} />
        </TabsContent>

        <TabsContent value="analytics" className="mt-6">
          <Card className="mb-4">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                Analyze distribution, trends, and statistics of SEC documents in the vector store.
              </div>
            </CardContent>
          </Card>
          <SECAnalytics />
        </TabsContent>

        <TabsContent value="visualization" className="mt-6">
          {selectedDocumentId ? (
            <>
              <Card className="mb-4">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Info className="h-4 w-4" />
                    Detailed chunk analysis showing how the document was processed and indexed.
                  </div>
                </CardContent>
              </Card>
              <SECChunkVisualization 
                documentId={selectedDocumentId}
                onBack={handleBackFromVisualization}
              />
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Layers className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Document Selected</h3>
                <p className="text-muted-foreground mb-4">
                  Select a document from the Document Library to view its chunk visualization
                </p>
                <div className="text-sm text-muted-foreground">
                  The chunk visualization shows how documents are processed, tokenized, and embedded for the vector store.
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SECDocumentsManager;
