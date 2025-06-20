## PART-1 : Document Chunking Strategy
For Document Chunking we tried different strategies. 

1) Fixed Chunking: A fixed size of 300 tokens with 48 tokens as overlap. The performance of the information extraction is good but sometimes context loss and incomplete sentences are chunked.

2) Semantic Chunking: In this chunking technique we broke down the whole dataset into chunk size of 300 tokens with an overlap of 5 sentence. This technique helped us avoid incomplete sentences being considered.

3) Hierarchical Chunking: In this chunking technique, the heading number and its title with the content before the next heading was found. This chunking technique works well as we can have a chunk of data of a particular topics intact. But the problem is the pattern can cause wrong prediction of number and title as the next topic thereby loss of context

4) Custom Chunking: In this chunking technique, the information from the pdf is extracted and stored in form of markdown where we understand the heading and subheading based on the boldness and size of the text.
Then in that section break the data into paragraphs. This worked well but needs a lot of cleaning of text data as the data contains text things with images and the text are jumbled because of the presence of image (Geometric shapes).

Conclusion: After some amount of analysis, I have observed that custom chunking can be a good chunking approach given that we have a good understanding of the type of document we would get. Also we need to analyse and see if the patterns are same or not. 
I choose Semantic Chunking as the results were better with semantic search approach because the number of sentences in each paragraph is small and having an overlap of 5 sentences would helpful to get the context clearer.
With each type of chunking technique, we uploaded the data on Vector database (Pinecone) and created few test cases. Then we compared the ground truth with the answer generated with respect to the context data given to LLM. We used similarity score to compare similarity.  


## PART-2 : Vector Database Implementation

1) For embedding of data chunks, we selected OpenAI's "text-embedding-ada-002" model. It is fast and affordable, embedding length is 1536 to understand lengthy context. No hustle of downloading and maintaining of opensource models. 

2) Choose Pinecone vector Database as it is esy to use, well documented and donot have to take care of the maintenance of the servers when using Opensource vector DB like Qdrant. 
If the results need to be very fast then we can have a self hosted vector DB else using Pinecone is userfriendly and makes reranking implementation easy.

## PART-3 : Retrieval Evaluation
For evaluating the retrieval's performance, we created dataset with the ground truth and compared it with the answers generated using LLM. This would give us a score on how similar the ground truth and the generated answer using data reference from pinecone is.
11 qa was compared and an average of similairty score was taken.
The qa contained a mix of easy, medium and hard questions.

Summary of average cosine similarity scores:
semantic       : 0.8876
fixed          : 0.8677
hierarchical   : 0.8495
custom         : 0.8459


The problem with the calculation is there are cases where the answer is correct but because the content is not highly similar the scores were less


## PART-4 : ReRanking

We choose Cross-encoder re-ranking as this is the best Reranking technique when we have take the most relevant information with respect to query.

The Reranking information + generated data is around three times slower when compared to normal information + generated data. But the number of reference required is 70% less for reranked + Generated data as compared to normal information + generated data


## PART-5 : Pipeline

Successfully built a basic pipeline that would read qeustions from .json file and answer questions by taking external informationa about the data from the vectordatabase.

This is then compared with the ground truth which is also provided and also checks for hallucination by looking into the source and validating if the answer given has text from source or not.
