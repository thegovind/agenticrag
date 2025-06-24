const API_BASE_URL = import.meta.env.MODE === 'production' 
  ? '/api/v1' 
  : 'http://localhost:8000/api/v1';

export interface DocumentUploadRequest {
  files: File[];
  embedding_model?: string;
  search_type?: string;
  temperature?: number;
  document_type?: string;
  company_name?: string;
  filing_date?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  processing_started_at: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  type: string;
  size: number;
  uploadDate: string;
  status: string;
  chunks?: number;
  conflicts?: number;
  processingProgress?: number;
}

export interface ConflictInfo {
  id: string;
  documentId: string;
  chunkId: string;
  conflictType: string;
  description: string;
  sources: string[];
  status: string;
}

export interface KnowledgeBaseMetrics {
  total_documents: number;
  total_chunks: number;
  active_conflicts: number;
  processing_rate: number;
  documents_by_type: Record<string, number>;
  processing_queue_size: number;
  last_updated: string;
}

class ApiService {
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async uploadDocuments(request: DocumentUploadRequest): Promise<DocumentUploadResponse[]> {
    const formData = new FormData();
    
    request.files.forEach(file => {
      formData.append('files', file);
    });
    
    if (request.embedding_model) {
      formData.append('embedding_model', request.embedding_model);
    }
    if (request.search_type) {
      formData.append('search_type', request.search_type);
    }
    if (request.temperature !== undefined) {
      formData.append('temperature', request.temperature.toString());
    }
    if (request.document_type) {
      formData.append('document_type', request.document_type);
    }
    if (request.company_name) {
      formData.append('company_name', request.company_name);
    }
    if (request.filing_date) {
      formData.append('filing_date', request.filing_date);
    }

    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Upload failed: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async listDocuments(params?: {
    document_type?: string;
    company_name?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ documents: DocumentInfo[] }> {
    const searchParams = new URLSearchParams();
    
    if (params?.document_type) searchParams.append('document_type', params.document_type);
    if (params?.company_name) searchParams.append('company_name', params.company_name);
    if (params?.status) searchParams.append('status', params.status);
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.offset) searchParams.append('offset', params.offset.toString());

    const queryString = searchParams.toString();
    const endpoint = queryString ? `/knowledge-base/documents?${queryString}` : '/knowledge-base/documents';
    
    return this.makeRequest<{ documents: DocumentInfo[] }>(endpoint);
  }

  async getDocument(documentId: string): Promise<DocumentInfo> {
    return this.makeRequest<DocumentInfo>(`/documents/${documentId}`);
  }

  async deleteDocument(documentId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  async reprocessDocument(documentId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/documents/${documentId}/reprocess`, {
      method: 'POST',
    });
  }

  async getConflicts(params?: {
    status?: string;
    document_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ conflicts: ConflictInfo[] }> {
    const searchParams = new URLSearchParams();
    
    if (params?.status) searchParams.append('status', params.status);
    if (params?.document_id) searchParams.append('document_id', params.document_id);
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.offset) searchParams.append('offset', params.offset.toString());

    const queryString = searchParams.toString();
    const endpoint = queryString ? `/knowledge-base/conflicts?${queryString}` : '/knowledge-base/conflicts';
    
    return this.makeRequest<{ conflicts: ConflictInfo[] }>(endpoint);
  }

  async resolveConflict(conflictId: string, status: 'resolved' | 'ignored'): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/knowledge-base/conflicts/${conflictId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
  }

  async getKnowledgeBaseMetrics(): Promise<KnowledgeBaseMetrics> {
    return this.makeRequest<KnowledgeBaseMetrics>('/knowledge-base/metrics');
  }

  async getDocumentContent(documentId: string, section?: string): Promise<{
    document_id: string;
    section?: string;
    content: string;
    chunks: any[];
    metadata: Record<string, any>;
  }> {
    const endpoint = section 
      ? `/documents/${documentId}/content?section=${encodeURIComponent(section)}`
      : `/documents/${documentId}/content`;
    
    return this.makeRequest(endpoint);
  }

  async getDocumentChunks(documentId: string, params?: {
    limit?: number;
    offset?: number;
    section?: string;
  }): Promise<{
    document_id: string;
    total_chunks: number;
    chunks: any[];
    section_filter?: string;
  }> {
    const searchParams = new URLSearchParams();
    
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.offset) searchParams.append('offset', params.offset.toString());
    if (params?.section) searchParams.append('section', params.section);

    const queryString = searchParams.toString();
    const endpoint = queryString 
      ? `/documents/${documentId}/chunks?${queryString}` 
      : `/documents/${documentId}/chunks`;
    
    return this.makeRequest(endpoint);
  }  async askQuestion(request: {
    question: string;
    session_id: string;
    verification_level: 'basic' | 'thorough' | 'comprehensive';
    chat_model?: string;
    embedding_model?: string;
    temperature?: number;
    credibility_check_enabled?: boolean;
  }): Promise<{
    answer: string;
    session_id: string;
    confidence_score: number;
    citations: any[];
    sub_questions: string[];
    verification_details: any;
    metadata: any;
    token_usage: any;
  }> {
    return this.makeRequest('/qa/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async decomposeQuestion(request: {
    question: string;
    session_id: string;
  }): Promise<{
    sub_questions: string[];
    session_id: string;
    metadata: any;
  }> {
    return this.makeRequest('/qa/decompose', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async verifySource(request: {
    source_url: string;
    content: string;
    session_id: string;
  }): Promise<{
    credibility_score: number;
    verification_details: any;
    session_id: string;
    metadata: any;
  }> {
    return this.makeRequest('/qa/verify-sources', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getQACapabilities(): Promise<{
    available_models: string[];
    verification_levels: string[];
    supported_features: string[];
    agent_status: any;
  }> {
    return this.makeRequest('/qa/capabilities');
  }
}

export const apiService = new ApiService();
