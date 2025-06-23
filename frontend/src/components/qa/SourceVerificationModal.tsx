import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { CheckCircle, AlertTriangle, XCircle, ExternalLink } from 'lucide-react';
import { VerifiedSource } from './QAContainer';

interface SourceVerificationProps {
  verifiedSources: VerifiedSource[];
  isOpen: boolean;
  onClose: () => void;
}

export const SourceVerification: React.FC<SourceVerificationProps> = ({
  verifiedSources,
  isOpen,
  onClose,
}) => {
  const getVerificationIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'questionable':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unverified':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-600" />;
    }
  };

  const getVerificationColor = (status: string) => {
    switch (status) {
      case 'verified':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800';
      case 'questionable':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300 dark:border-yellow-800';
      case 'unverified':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-700';
    }
  };

  // Helper function to safely format credibility scores and avoid NaN
  const formatCredibilityScore = (score: number | undefined): string => {
    if (score === undefined || score === null || isNaN(score)) {
      return '0';
    }
    return (score * 100).toFixed(0);
  };

  const getCredibilityColor = (score: number | undefined) => {
    const safeScore = score || 0;
    if (safeScore >= 0.7) return 'text-green-600';
    if (safeScore >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getCredibilityLabel = (score: number | undefined) => {
    const safeScore = score || 0;
    if (safeScore >= 0.8) return 'Very High';
    if (safeScore >= 0.7) return 'High';
    if (safeScore >= 0.5) return 'Medium';
    if (safeScore >= 0.3) return 'Low';
    return 'Very Low';
  };

  // Calculate overall credibility from all sources
  const overallCredibility = verifiedSources.length > 0 
    ? verifiedSources.reduce((sum, source) => sum + (source.credibilityScore || 0), 0) / verifiedSources.length
    : 0;

  const verifiedCount = verifiedSources.filter(s => s.verificationStatus === 'verified').length;
  const questionableCount = verifiedSources.filter(s => s.verificationStatus === 'questionable').length;
  const unverifiedCount = verifiedSources.filter(s => s.verificationStatus === 'unverified').length;  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl h-[85vh] max-h-[85vh] flex flex-col">
        <DialogHeader className="flex-shrink-0 pb-4">
          <DialogTitle className="text-xl">Source Verification Results</DialogTitle>
          <div className="space-y-3 pt-2">
            <div className="flex items-center space-x-4 text-sm">
              <span className="font-medium">
                Overall Credibility: 
                <span className={`ml-1 ${getCredibilityColor(overallCredibility)}`}>
                  {formatCredibilityScore(overallCredibility)}% ({getCredibilityLabel(overallCredibility)})
                </span>
              </span>
            </div>
            <div className="flex items-center space-x-6 text-sm text-muted-foreground">
              <div className="flex items-center space-x-1">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span>{verifiedCount} Verified</span>
              </div>
              <div className="flex items-center space-x-1">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                <span>{questionableCount} Questionable</span>
              </div>
              <div className="flex items-center space-x-1">
                <XCircle className="h-4 w-4 text-red-600" />
                <span>{unverifiedCount} Unverified</span>
              </div>
            </div>
          </div>
        </DialogHeader>        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full max-h-[60vh] pr-2 scrollbar-enhanced">
            <div className="space-y-4 pb-2 pr-2">
              {verifiedSources.map((source) => (
              <div key={source.sourceId} className="border rounded-lg p-4 space-y-3 bg-card">
                <div className="flex items-start justify-between">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center space-x-2">
                      {getVerificationIcon(source.verificationStatus)}
                      <h4 className="font-medium text-base text-foreground">
                        {source.title}
                      </h4>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge variant="outline" className={getVerificationColor(source.verificationStatus)}>
                        {source.verificationStatus}
                      </Badge>
                      <span className={`text-sm font-medium ${getCredibilityColor(source.credibilityScore)}`}>
                        {formatCredibilityScore(source.credibilityScore)}% credible
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {source.content.length > 200 
                      ? source.content.substring(0, 200) + '...'
                      : source.content
                    }
                  </p>
                  
                  {source.url && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs h-7"
                      onClick={() => window.open(source.url, '_blank')}
                    >
                      <ExternalLink className="h-3 w-3 mr-1" />
                      View Source
                    </Button>
                  )}
                </div>

                <div className="border-t pt-3 space-y-2">
                  <div className="text-sm">
                    <span className="font-medium text-foreground">Credibility Analysis:</span>
                    <p className="text-muted-foreground mt-1">
                      {source.credibilityExplanation}
                    </p>
                  </div>

                  {source.trustIndicators.length > 0 && (
                    <div className="text-sm">
                      <span className="font-medium text-green-700 dark:text-green-400">Trust Indicators:</span>
                      <ul className="list-disc list-inside text-muted-foreground mt-1 space-y-0.5">
                        {source.trustIndicators.map((indicator, index) => (
                          <li key={index} className="text-xs">{indicator}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {source.redFlags.length > 0 && (
                    <div className="text-sm">
                      <span className="font-medium text-red-700 dark:text-red-400">Red Flags:</span>
                      <ul className="list-disc list-inside text-muted-foreground mt-1 space-y-0.5">
                        {source.redFlags.map((flag, index) => (
                          <li key={index} className="text-xs">{flag}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {verifiedSources.length === 0 && (
              <div className="text-center py-8">
                <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                <p className="text-muted-foreground">No verified sources available</p>
              </div>              )}
            </div>
            <ScrollBar />
          </ScrollArea>
        </div>
      </DialogContent>
    </Dialog>
  );
};
