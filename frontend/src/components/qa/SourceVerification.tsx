import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { X, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';
import { VerifiedSource } from './QAContainer';

interface SourceVerificationProps {
  verifiedSources: VerifiedSource[];
  onClose: () => void;
}

export const SourceVerification: React.FC<SourceVerificationProps> = ({
  verifiedSources,
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

  const getCredibilityColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getCredibilityLabel = (score: number) => {
    if (score >= 0.8) return 'Highly Credible';
    if (score >= 0.6) return 'Credible';
    if (score >= 0.4) return 'Moderately Credible';
    return 'Low Credibility';
  };

  const overallCredibility = verifiedSources.length > 0
    ? verifiedSources.reduce((sum, source) => sum + source.credibilityScore, 0) / verifiedSources.length
    : 0;

  const verifiedCount = verifiedSources.filter(s => s.verificationStatus === 'verified').length;
  const questionableCount = verifiedSources.filter(s => s.verificationStatus === 'questionable').length;
  const unverifiedCount = verifiedSources.filter(s => s.verificationStatus === 'unverified').length;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Source Verification</CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="space-y-2">
          <div className="flex items-center space-x-4 text-sm">
            <span className="font-medium">
              Overall Credibility: 
              <span className={`ml-1 ${getCredibilityColor(overallCredibility)}`}>
                {(overallCredibility * 100).toFixed(0)}% ({getCredibilityLabel(overallCredibility)})
              </span>
            </span>
          </div>
          <div className="flex items-center space-x-4 text-sm text-muted-foreground">
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-3 w-3 text-green-600" />
              <span>{verifiedCount} Verified</span>
            </div>
            <div className="flex items-center space-x-1">
              <AlertTriangle className="h-3 w-3 text-yellow-600" />
              <span>{questionableCount} Questionable</span>
            </div>
            <div className="flex items-center space-x-1">
              <XCircle className="h-3 w-3 text-red-600" />
              <span>{unverifiedCount} Unverified</span>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="space-y-4">
            {verifiedSources.map((source) => (
              <div key={source.sourceId} className="border rounded-lg p-3 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center space-x-2">
                      {getVerificationIcon(source.verificationStatus)}
                      <h4 className="font-medium text-sm text-foreground line-clamp-1">
                        {source.title}
                      </h4>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className={getVerificationColor(source.verificationStatus)}>
                        {source.verificationStatus}
                      </Badge>
                      <span className={`text-sm font-medium ${getCredibilityColor(source.credibilityScore)}`}>
                        {(source.credibilityScore * 100).toFixed(0)}% credible
                      </span>
                    </div>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground bg-muted/30 p-2 rounded">
                  {source.content.length > 200 
                    ? source.content.substring(0, 200) + '...'
                    : source.content
                  }
                </div>

                <div className="space-y-2">
                  <div className="text-xs">
                    <strong className="text-foreground">Credibility Assessment:</strong>
                    <p className="text-muted-foreground mt-1">{source.credibilityExplanation}</p>
                  </div>

                  {source.trustIndicators.length > 0 && (
                    <div className="text-xs">
                      <strong className="text-green-700">Trust Indicators:</strong>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {source.trustIndicators.map((indicator, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800">
                            {indicator}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {source.redFlags.length > 0 && (
                    <div className="text-xs">
                      <strong className="text-red-700">Red Flags:</strong>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {source.redFlags.map((flag, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800">
                            {flag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {source.url && (
                  <div className="pt-2 border-t">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline"
                    >
                      View Original Source
                    </a>
                  </div>
                )}
              </div>
            ))}

            {verifiedSources.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                <p>No source verification data available</p>
                <p className="text-xs mt-1">Try clicking "Verify Sources" on an answer to see detailed credibility analysis</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
