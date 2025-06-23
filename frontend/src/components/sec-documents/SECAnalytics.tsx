import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { 
  BarChart3, 
  TrendingUp, 
  FileText, 
  Building, 
  Calendar, 
  RefreshCw,
  Activity,
  PieChart,
  Users
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface FormTypeDistribution {
  [key: string]: number;
}

interface CompanyDistribution {
  [key: string]: number;
}

interface RecentActivity {
  document_id: string;
  company: string;
  form_type: string;
  filing_date: string;
  processed_at: string;
  chunk_count: number;
}

interface SECAnalyticsData {
  total_documents: number;
  total_chunks: number;
  companies_count: number;
  form_types_distribution: FormTypeDistribution;
  chunks_per_document_avg: number;
  recent_activity: RecentActivity[];
  company_distribution: CompanyDistribution;
  filing_date_range: {
    earliest?: string;
    latest?: string;
  };
}

const SECAnalytics: React.FC = () => {
  const [analytics, setAnalytics] = useState<SECAnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadAnalytics();
  }, []);
  const loadAnalytics = async () => {
    setLoading(true);
    try {
      console.log('Loading SEC analytics...');
      const response = await fetch('/api/v1/sec/analytics');
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Analytics API error:', response.status, errorText);
        throw new Error(`Failed to load analytics: ${response.status}`);
      }

      const data: SECAnalyticsData = await response.json();
      console.log('Received analytics data:', data);
      setAnalytics(data);

    } catch (error) {
      console.error('Error loading analytics:', error);
      toast({
        title: "Error",
        description: `Failed to load analytics data: ${error}`,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString();
  };

  const getTopItems = (distribution: { [key: string]: number }, limit: number = 5) => {
    return Object.entries(distribution)
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit);
  };

  const calculatePercentage = (value: number, total: number) => {
    return total > 0 ? ((value / total) * 100).toFixed(1) : '0';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!analytics) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <div className="text-muted-foreground">No analytics data available</div>
          <Button onClick={loadAnalytics} className="mt-4">
            Load Analytics
          </Button>
        </CardContent>
      </Card>
    );
  }

  const topFormTypes = getTopItems(analytics.form_types_distribution);
  const topCompanies = getTopItems(analytics.company_distribution);
  const totalFormTypeChunks = Object.values(analytics.form_types_distribution).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            SEC Document Analytics
          </CardTitle>
          <CardDescription>
            Insights and statistics about SEC documents in the vector store
          </CardDescription>
          <Button 
            onClick={loadAnalytics} 
            disabled={loading} 
            className="w-fit gap-2"
            variant="outline"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </CardHeader>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Documents</p>
                <p className="text-3xl font-bold text-blue-600">{analytics.total_documents}</p>
              </div>
              <FileText className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Chunks</p>
                <p className="text-3xl font-bold text-green-600">{analytics.total_chunks}</p>
              </div>
              <Activity className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Companies</p>
                <p className="text-3xl font-bold text-purple-600">{analytics.companies_count}</p>
              </div>
              <Users className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Chunks/Doc</p>
                <p className="text-3xl font-bold text-orange-600">{analytics.chunks_per_document_avg}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Date Range */}
      {analytics.filing_date_range.earliest && analytics.filing_date_range.latest && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Filing Date Range
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Earliest Filing</p>
                <p className="text-lg font-semibold">{formatDate(analytics.filing_date_range.earliest)}</p>
              </div>
              <div className="text-center">
                <div className="h-1 w-32 bg-gradient-to-r from-blue-500 to-green-500 rounded"></div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Latest Filing</p>
                <p className="text-lg font-semibold">{formatDate(analytics.filing_date_range.latest)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form Types Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5" />
              Form Types Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topFormTypes.map(([formType, count]) => (
                <div key={formType} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{formType}</span>
                    <span className="text-muted-foreground">
                      {count} chunks ({calculatePercentage(count, totalFormTypeChunks)}%)
                    </span>
                  </div>
                  <Progress 
                    value={parseFloat(calculatePercentage(count, totalFormTypeChunks))} 
                    className="h-2" 
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Companies */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Building className="h-5 w-5" />
              Top Companies by Chunks
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topCompanies.map(([company, count]) => (
                <div key={company} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium truncate" title={company}>
                      {company.length > 30 ? `${company.substring(0, 30)}...` : company}
                    </span>
                    <span className="text-muted-foreground">
                      {count} chunks
                    </span>
                  </div>
                  <Progress 
                    value={(count / Math.max(...Object.values(analytics.company_distribution))) * 100} 
                    className="h-2" 
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Processing Activity</CardTitle>
          <CardDescription>Latest processed SEC documents</CardDescription>
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
                </TableRow>
              </TableHeader>
              <TableBody>
                {analytics.recent_activity.map((activity) => (
                  <TableRow key={activity.document_id}>
                    <TableCell className="font-medium">{activity.company}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{activity.form_type}</Badge>
                    </TableCell>
                    <TableCell>{formatDate(activity.filing_date)}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{activity.chunk_count}</Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(activity.processed_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          
          {analytics.recent_activity.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              No recent activity found
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SECAnalytics;
