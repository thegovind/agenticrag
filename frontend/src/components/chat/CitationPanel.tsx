import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { X, ExternalLink, FileText, Eye, Download } from 'lucide-react';
import { Citation } from './ChatContainer';

interface CitationPanelProps {
  citations: Citation[];
  onClose: () => void;
}

export const CitationPanel: React.FC<CitationPanelProps> = ({
  citations,
  onClose,
}) => {
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(
    citations.length > 0 ? citations[0] : null
  );

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const handleViewDocument = (citation: Citation) => {
    if (citation.url) {
      window.open(citation.url, '_blank');
    } else {
      console.log('View document:', citation.documentId);
    }
  };

  const handleDownloadDocument = (citation: Citation) => {
    console.log('Download document:', citation.documentId);
  };

  return (
    <div className="h-full flex flex-col bg-background border-l">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          <h3 className="font-semibold">Sources ({citations.length})</h3>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="list" className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-2 m-4 mb-0">
            <TabsTrigger value="list">Source List</TabsTrigger>
            <TabsTrigger value="detail">Detail View</TabsTrigger>
          </TabsList>

          <TabsContent value="list" className="flex-1 m-0 p-4">
            <ScrollArea className="h-full">
              <div className="space-y-3">
                {citations.map((citation, index) => (
                  <Card
                    key={citation.id}
                    className={`p-4 cursor-pointer transition-colors hover:bg-muted/50 ${
                      selectedCitation?.id === citation.id ? 'ring-2 ring-primary' : ''
                    }`}
                    onClick={() => setSelectedCitation(citation)}
                  >
                    <div className="space-y-2">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="font-medium text-sm leading-tight">
                            {citation.documentTitle}
                          </h4>
                          {citation.sectionTitle && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {citation.sectionTitle}
                            </p>
                          )}
                        </div>
                        <Badge
                          className={`text-xs ${getConfidenceColor(citation.confidence)}`}
                        >
                          {citation.confidence}
                        </Badge>
                      </div>

                      <p className="text-xs text-muted-foreground line-clamp-3">
                        {citation.content}
                      </p>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>#{index + 1}</span>
                          {citation.pageNumber && (
                            <>
                              <Separator orientation="vertical" className="h-3" />
                              <span>Page {citation.pageNumber}</span>
                            </>
                          )}
                        </div>

                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleViewDocument(citation);
                            }}
                            className="h-6 w-6 p-0"
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownloadDocument(citation);
                            }}
                            className="h-6 w-6 p-0"
                          >
                            <Download className="h-3 w-3" />
                          </Button>
                          {citation.url && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open(citation.url, '_blank');
                              }}
                              className="h-6 w-6 p-0"
                            >
                              <ExternalLink className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="detail" className="flex-1 m-0 p-4">
            {selectedCitation ? (
              <ScrollArea className="h-full">
                <div className="space-y-4">
                  <Card className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <h3 className="font-semibold">{selectedCitation.documentTitle}</h3>
                        <Badge
                          className={`${getConfidenceColor(selectedCitation.confidence)}`}
                        >
                          {selectedCitation.confidence} confidence
                        </Badge>
                      </div>

                      {selectedCitation.sectionTitle && (
                        <div>
                          <Label className="text-xs font-medium text-muted-foreground">
                            Section
                          </Label>
                          <p className="text-sm">{selectedCitation.sectionTitle}</p>
                        </div>
                      )}

                      {selectedCitation.pageNumber && (
                        <div>
                          <Label className="text-xs font-medium text-muted-foreground">
                            Page Number
                          </Label>
                          <p className="text-sm">{selectedCitation.pageNumber}</p>
                        </div>
                      )}

                      <div>
                        <Label className="text-xs font-medium text-muted-foreground">
                          Source
                        </Label>
                        <p className="text-sm">{selectedCitation.source}</p>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <div className="space-y-2">
                      <Label className="text-xs font-medium text-muted-foreground">
                        Relevant Content
                      </Label>
                      <div className="prose prose-sm max-w-none">
                        <p className="text-sm leading-relaxed">
                          {selectedCitation.content}
                        </p>
                      </div>
                    </div>
                  </Card>

                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleViewDocument(selectedCitation)}
                      className="flex-1"
                    >
                      <Eye className="h-4 w-4 mr-2" />
                      View Document
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleDownloadDocument(selectedCitation)}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    {selectedCitation.url && (
                      <Button
                        variant="outline"
                        onClick={() => window.open(selectedCitation.url, '_blank')}
                      >
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </ScrollArea>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                Select a citation to view details
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
