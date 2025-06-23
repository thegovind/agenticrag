/**
 * Azure OpenAI Deployment Service
 * 
 * Service for fetching available Azure OpenAI deployments for model configuration
 */

export interface DeploymentInfo {
  deployment_name: string;
  model_name: string;
  model_version: string;
  sku: string;
  capacity: number;
  state: string;
  display_name: string;
}

export interface DeploymentResponse {
  success: boolean;
  data: DeploymentInfo[];
}

export interface DeploymentSummary {
  total_deployments: number;
  chat_models: DeploymentInfo[];
  embedding_models: DeploymentInfo[];
  account_name: string;
  resource_group: string;
  subscription_id: string;
}

export interface DeploymentSummaryResponse {
  success: boolean;
  data: DeploymentSummary;
}

class AzureOpenAIDeploymentService {
  private baseUrl: string;

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  async getAllDeployments(): Promise<DeploymentSummary> {
    try {
      const response = await fetch(`${this.baseUrl}/deployments/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result: DeploymentSummaryResponse = await response.json();
      
      if (!result.success) {
        throw new Error('Failed to fetch deployments');
      }
      
      return result.data;
    } catch (error) {
      console.error('Error fetching all deployments:', error);
      throw error;
    }
  }

  async getChatDeployments(): Promise<DeploymentInfo[]> {
    try {
      const response = await fetch(`${this.baseUrl}/deployments/chat`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result: DeploymentResponse = await response.json();
      
      if (!result.success) {
        throw new Error('Failed to fetch chat deployments');
      }
      
      return result.data;
    } catch (error) {
      console.error('Error fetching chat deployments:', error);
      throw error;
    }
  }

  async getEmbeddingDeployments(): Promise<DeploymentInfo[]> {
    try {
      const response = await fetch(`${this.baseUrl}/deployments/embedding`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result: DeploymentResponse = await response.json();
      
      if (!result.success) {
        throw new Error('Failed to fetch embedding deployments');
      }
      
      return result.data;
    } catch (error) {
      console.error('Error fetching embedding deployments:', error);
      throw error;
    }
  }

  async checkHealth(): Promise<{ status: string; account_name: string; total_deployments: number; service_type: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/deployments/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      
      if (!result.success) {
        throw new Error('Deployment service unhealthy');
      }
      
      return result;
    } catch (error) {
      console.error('Error checking deployment service health:', error);
      throw error;
    }
  }
}

export const deploymentService = new AzureOpenAIDeploymentService();
