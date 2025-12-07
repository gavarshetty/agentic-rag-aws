import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3_notifications from 'aws-cdk-lib/aws-s3-notifications';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as bedrock from 'aws-cdk-lib/aws-bedrock';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as apigatewayv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as apigatewayv2_integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';

/**
 * The AgenticRagStack creates all necessary AWS resources for the Agentic RAG application:
 * - S3 bucket for knowledge base source files
 * - OpenSearch Serverless collection for vector database
 * - Bedrock Knowledge Base with vector storage
 * - Document ingest Lambda function triggered by S3 uploads
 * - RAG Lambda function for retrieval and generation
 * - API Gateway HTTP API for frontend integration
 * - Comprehensive IAM roles with least privilege permissions
 */
export class AgenticRagStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================
    // S3 BUCKET FOR KNOWLEDGE BASE SOURCE FILES
    // ========================================
    // S3 bucket to store uploaded documents for knowledge base ingestion
    const knowledgeBaseBucket = new s3.Bucket(this, 'AgenticRagKnowledgeBaseBucket', {
      bucketName: `agentic-rag-kb-${this.account}-${this.region}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'DeleteOldVersions',
          enabled: true,
          noncurrentVersionExpiration: cdk.Duration.days(30),
          expiration: cdk.Duration.days(30),
        },
      ],
    });

    // ========================================
    // OPEN SEARCH SERVERLESS COLLECTION
    // ========================================
    // Vector database collection for storing document embeddings and metadata
    const openSearchCollection = new opensearchserverless.CfnCollection(this, 'AgenticRagOpenSearchCollection', {
      name: 'agentic-rag-collection',
      type: 'VECTORSEARCH',
      description: 'Vector database for Agentic RAG knowledge base',
    });


    // ========================================
    // BEDROCK KNOWLEDGE BASE
    // ========================================

    // IAM role granting Bedrock access to OpenSearch and S3 for knowledge base operations
    const knowledgeBaseRole = new iam.Role(this, 'AgenticRagKnowledgeBaseRole', {
      roleName: 'agentic-rag-kb-role',
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    });

    // Policy allowing Bedrock to manage OpenSearch indexes and documents
    knowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'aoss:APIAccessAll',
        'aoss:CreateIndex',
        'aoss:DeleteIndex',
        'aoss:UpdateIndex',
        'aoss:DescribeIndex',
        'aoss:ReadDocument',
        'aoss:WriteDocument',
      ],
      resources: [openSearchCollection.attrArn],
    }));

    // Policy allowing Bedrock to read documents from S3 bucket
    knowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:ListBucket',
      ],
      resources: [
        knowledgeBaseBucket.bucketArn,
        `${knowledgeBaseBucket.bucketArn}/*`,
      ],
    }));


    // Bedrock knowledge base that integrates Titan embeddings with OpenSearch vector storage
    const knowledgeBase = new bedrock.CfnKnowledgeBase(this, 'AgenticRagKnowledgeBase', {
      name: 'agentic-rag-knowledge-base',
      description: 'Knowledge base for Agentic RAG application',
      roleArn: knowledgeBaseRole.roleArn, 
      knowledgeBaseConfiguration: {
        type: 'VECTOR',
        vectorKnowledgeBaseConfiguration: {
          embeddingModelArn: `arn:aws:bedrock:${this.region}::foundation-model/amazon.titan-embed-text-v1`,
        },
      },
      storageConfiguration: {
        type: 'OPENSEARCH_SERVERLESS',
        opensearchServerlessConfiguration: {
          collectionArn: openSearchCollection.attrArn,
          vectorIndexName: 'agentic-rag-index',
          fieldMapping: {
            vectorField: 'vector',
            textField: 'text',
            metadataField: 'metadata',
          },
        },
      },
    });

    // Data source definition connecting the S3 documents/ folder to the Bedrock knowledge base
    const dataSource = new bedrock.CfnDataSource(this, 'DocumentDataSource', {
      knowledgeBaseId: knowledgeBase.attrKnowledgeBaseId,
      name: 'agintic-rag-s3-data-source',
      dataSourceConfiguration: {
        type: 'S3',
        s3Configuration: {
          bucketArn: knowledgeBaseBucket.bucketArn,
          inclusionPrefixes: ['documents/']  // Only files in documents/ folder
        }
      }
    });
    
    
    // ========================================
    // LAMBDA FUNCTIONS
    // ========================================
    
    // IAM role for Lambda function that processes uploaded documents and updates knowledge base
    const ingestLambdaRole = new iam.Role(this, 'AgenticRagIngestLambdaRole', {
      roleName: 'agentic-rag-ingest-lambda-role',
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    ingestLambdaRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'));

    // Policy allowing Lambda to read documents from S3 bucket for processing
    ingestLambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:GetObjectVersion',
        's3:ListBucket',
      ],
      resources: [
        knowledgeBaseBucket.bucketArn,
        `${knowledgeBaseBucket.bucketArn}/*`,
      ],
    }));

    // Policy allowing Lambda to manage THIS SPECIFIC knowledge base ingestion jobs only
    ingestLambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:StartIngestionJob',
        'bedrock:GetIngestionJob',
        'bedrock:ListIngestionJobs',
      ],
      resources: [knowledgeBase.attrKnowledgeBaseArn], // Only this specific KB
    }));

    // Policy allowing Lambda to invoke Titan embedding model for document processing
    ingestLambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
      ],
      resources: [`arn:aws:bedrock:${this.region}::foundation-model/amazon.titan-embed-text-v1`],
    }));

    // IAM role for Lambda function that handles RAG queries and generates responses
    const ragLambdaRole = new iam.Role(this, 'AgenticRagLambdaRole', {
      roleName: 'agentic-rag-lambda-role',
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    ragLambdaRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'));

    // Policy allowing Lambda to read documents from S3 bucket for context retrieval
    ragLambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:ListBucket',
      ],
      resources: [
        knowledgeBaseBucket.bucketArn,
        `${knowledgeBaseBucket.bucketArn}/*`,
      ],
    }));

    // Policy allowing Lambda to retrieve from THIS SPECIFIC knowledge base and invoke Claude/Llama models
    ragLambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:Retrieve',
        'bedrock:RetrieveAndGenerate',
        'bedrock:InvokeModel',
      ],
      resources: [
        knowledgeBase.attrKnowledgeBaseArn, 
        `arn:aws:bedrock:${this.region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0`,
        `arn:aws:bedrock:${this.region}::foundation-model/meta.llama3-1-8b-instruct-v1:0`,
      ],
    }));

    

    // Lambda function triggered by S3 uploads to process documents and update knowledge base
    const ingestLambda = new lambda.Function(this, 'AgenticRagIngestLambda', {
      functionName: 'agentic-rag-ingest-function',
      runtime: lambda.Runtime.PYTHON_3_12,
      code: lambda.Code.fromAsset('../backend/lambda_src'),
      handler: 'handlers.knowledge_base_handler.handler',
      role: ingestLambdaRole,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        KNOWLEDGE_BASE_ID: knowledgeBase.attrKnowledgeBaseId,
        S3_BUCKET_NAME: knowledgeBaseBucket.bucketName,
        S3_DATA_SOURCE_ID: dataSource.attrDataSourceId,
      },
    });

    // Lambda function that performs retrieval-augmented generation using Bedrock models
    const ragLambda = new lambda.Function(this, 'AgenticRagLambda', {
      functionName: 'agentic-rag-function',
      runtime: lambda.Runtime.PYTHON_3_12,
      code: lambda.Code.fromAsset('../backend/lambda_src'),
      handler: 'handlers.rag_handler.handler',
      role: ragLambdaRole,
      timeout: cdk.Duration.minutes(10),
      memorySize: 2048,
      environment: {
        KNOWLEDGE_BASE_ID: knowledgeBase.attrKnowledgeBaseId,
        S3_BUCKET_NAME: knowledgeBaseBucket.bucketName,
      },
    });

    // ========================================
    // S3 NOTIFICATIONS
    // ========================================
    // Trigger document ingest Lambda when PDF files are uploaded to documents/ folder
    knowledgeBaseBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3_notifications.LambdaDestination(ingestLambda),
      { prefix: 'documents/', suffix: '.pdf' } // Only trigger on PDF files in documents/ folder
    );

    // Trigger document ingest Lambda when TXT files are uploaded to documents/ folder
    knowledgeBaseBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3_notifications.LambdaDestination(ingestLambda),
      { prefix: 'documents/', suffix: '.txt' } // Also trigger on TXT files
    );

    // // ========================================
    // // API GATEWAY
    // // ========================================
    // // HTTP API Gateway to expose RAG functionality to frontend applications
    // const apiGateway = new apigatewayv2.HttpApi(this, 'AgenticRagApi', {
    //   apiName: 'agentic-rag-api',
    //   description: 'API Gateway for Agentic RAG application',
    //   corsPreflight: {
    //     allowHeaders: ['*'],
    //     allowMethods: [apigatewayv2.CorsHttpMethod.POST, apigatewayv2.CorsHttpMethod.GET],
    //     allowOrigins: ['*'],
    //   },
    // });

    // // API Gateway integration to connect HTTP requests to the RAG Lambda function
    // const ragIntegration = new apigatewayv2_integrations.HttpLambdaIntegration('AgenticRagIntegration', ragLambda);

    // // Route POST requests to /chat endpoint to the RAG Lambda function
    // apiGateway.addRoutes({
    //   path: '/chat',
    //   methods: [apigatewayv2.HttpMethod.POST],
    //   integration: ragIntegration,
    // });

    // ========================================
    // OUTPUTS
    // ========================================
    // S3 bucket name for uploading knowledge base documents
    new cdk.CfnOutput(this, 'KnowledgeBaseBucketName', {
      value: knowledgeBaseBucket.bucketName,
      description: 'S3 bucket for knowledge base source files',
    });

    // Unique identifier for the Bedrock knowledge base
    new cdk.CfnOutput(this, 'KnowledgeBaseId', {
      value: knowledgeBase.attrKnowledgeBaseId,
      description: 'Bedrock Knowledge Base ID',
    });

    // ARN of the OpenSearch Serverless vector collection
    new cdk.CfnOutput(this, 'OpenSearchCollectionArn', {
      value: openSearchCollection.attrArn,
      description: 'OpenSearch Serverless collection ARN',
    });

    // // HTTP endpoint URL for frontend API calls
    // new cdk.CfnOutput(this, 'ApiGatewayUrl', {
    //   value: apiGateway.apiEndpoint,
    //   description: 'API Gateway endpoint URL',
    // });

    // ARN of the Lambda function that processes document uploads
    new cdk.CfnOutput(this, 'IngestLambdaArn', {
      value: ingestLambda.functionArn,
      description: 'Document ingest Lambda function ARN',
    });

    // ARN of the Lambda function that handles RAG queries
    new cdk.CfnOutput(this, 'RagLambdaArn', {
      value: ragLambda.functionArn,
      description: 'RAG Lambda function ARN',
    });
  }
}
