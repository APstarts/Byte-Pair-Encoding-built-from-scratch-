# Initialize the tokenizer
```tokenizer = BytePairTokenizer(vocab_size=1000)```

# Train once
``` tokenizer.train(text) ```

# Encode
``` ids = tokenizer.encode("hello world") ```

# Decode
``` text = tokenizer.decode(ids) ```