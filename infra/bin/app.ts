#!/usr/bin/env node
import * as dotenv from 'dotenv'; 
import * as cdk from 'aws-cdk-lib';
import { AgenticRagStack } from '../lib/stacks/agentic-rag-stack';

const app = new cdk.App();

dotenv.config();

new AgenticRagStack(app, 'AgenticRagStack', {
  env: {
    account: process.env.AWS_ACCOUNT_ID,
    region: process.env.AWS_REGION,
  },
});
