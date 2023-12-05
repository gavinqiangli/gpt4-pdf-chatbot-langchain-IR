import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

import { kv } from '@vercel/kv';

/*
Vercel Serverless Functions have the following defaults and maximum limits for the duration of a function:
For Hobby it is 10s.
If a Serverless Function doesn't respond within 10s duration, a 504 error code (FUNCTION_INVOCATION_TIMEOUT) is returned.
Functions using the Edge Runtime do not have a maximum duration. They must begin sending a response within 25 seconds and can continue streaming a response beyond that time.
2023-Oct-2 Most libraries which use Node.js APIs as dependencies can't be used in Edge Functions yet, but we're adding support for more Node.js APIs over time. See available APIs for a full list.
e.g. NextApiRequest and NextApiResponse are not supported. Only NextRequest and NextResponse from 'next/server' are supported.
*/
/*
export const config = {
  runtime: 'edge',
};
*/


export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { feedback, history } = req.body;

  console.log('feedback', feedback);

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!feedback) {
    return res.status(400).json({ message: 'No feedback in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedFeedback = feedback.trim().replaceAll('\n', ' ');

  try {
    
    /* We can  construct an LLMChain from openAI.
    const Feedback_PROMPT = `You are intelligent AI robot can learn and improve by yourself. Use the following conversation and user feedback to reflect yourself and improve your future answers to similar questions based on user feedback.
     Can you describe what have you learnt in 500 words?
    
    Chat History:
    {chat_history}
    
    Feedback: {feedback}
    `; 
    
      const model = new OpenAI({
      temperature: 0, // increase temepreature to get more creative answers
      modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    }); */

    // We can also construct an LLMChain from a ChatPromptTemplate and a chat model.
    const chat = new ChatOpenAI({ temperature: 0 });
    const chatPrompt = ChatPromptTemplate.fromPromptMessages([
      SystemMessagePromptTemplate.fromTemplate(
        `You are intelligent AI robot can learn and improve by yourself. Use the following conversation with human feedback to reflect yourself and write 500 words to summarize what have you learnt from human feedback.
        
        Chat History:
        {chat_history}
        `
      ),
      HumanMessagePromptTemplate.fromTemplate("{human_feedback}"),
    ]);
    const chain = new LLMChain({
      prompt: chatPrompt,
      llm: chat,
    });

    // Here is the single bottleneck of 10s timeout, ChatOpenAI can take more than 10s to respond depends on complexity and length of human feedback...
    const response = await chain.call({
      chat_history: history || [],
      human_feedback: sanitizedFeedback,
    });
    console.log({ response });

    // learn about the feedback and reflection by generatting new embeddings and vectors
    const fbDocs = 'This is user feedback:' + '\n' + sanitizedFeedback + '\n' + 'This is my reflection:' + '\n' + response.text;
    console.log({ fbDocs });

    // save raw data into feedback database, using vercel KV database
    async function updatefbdata() {
      try {
        await kv.append('fbdata', '\n'+ fbDocs);
        console.log('fbdata saved successfully:', fbDocs);
      } catch (error) {
        console.error('fbdata saving:', error);
      }
    }
    updatefbdata();

    // get the latest feedback data from database
    /* only for testing purpose. not needed for runtime. (Vercel serverless app has only 10s timeout to return 504 error...)
    async function getfbdata() {
      try {
        const fbdata = await kv.get('fbdata');
        console.log('fbdata retrieved successfully:', fbdata || {});
      } catch (error) {
        console.error('fbdata retrieving:', error);
      }
    }
    getfbdata(); */

    // RLHF learning process based on human feedback and self reflection
    async function learnfbdata() {
      // Split text into chunks
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 100,
        chunkOverlap: 0,
      });
      const rawDocs = await textSplitter.createDocuments([fbDocs]);
      const docs = await textSplitter.splitDocuments(rawDocs);
      console.log('split docs', docs);

      console.log('creating vector store...');
      // create and store the embeddings in the vectorStore
      const embeddings = new OpenAIEmbeddings();
      const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name

      //embed the PDF documents
      await PineconeStore.fromDocuments(docs, embeddings, {
        pineconeIndex: index,
        namespace: PINECONE_NAME_SPACE,
        textKey: 'text',
      });
    }
    learnfbdata();

    // Must respond within 10s, as Vercel serverless app has only 10s timeout to return 504 error...
    console.log('response', response);
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Failed to ingest your feedback' });
  }
}
