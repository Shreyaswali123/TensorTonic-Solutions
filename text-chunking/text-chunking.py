def text_chunking(tokens, chunk_size, overlap):
    """
    Splits a list of tokens into chunks with a specified overlap.
    
    Args:
        tokens (list): List of string tokens.
        chunk_size (int): The target size for each chunk.
        overlap (int): The number of tokens to overlap between chunks.
        
    Returns:
        list[list]: A list of token chunks.
    """
    if not tokens:
        return []
    
    # Step size determines the start of the next chunk
    step = chunk_size - overlap
    chunks = []
    
    # Iterate through the list starting from 0, jumping by 'step'
    for i in range(0, len(tokens), step):
        # Extract the chunk from the current start index
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop condition: if the current chunk reached or passed the end of the list
        if i + chunk_size >= len(tokens):
            break
            
    return chunks