# Initialize the tokenizer
```tokenizer = BytePairTokenizer(vocab_size=1000)```

# Train once
``` tokenizer.train(text) ```

# Encode
``` ids = tokenizer.encode("hello world") ```

# Decode
``` text = tokenizer.decode(ids) ```

# How to use the save functionality
``` tokenizer.save("tokenizer.pkl") ```

# How to use the load functionality
```tokenizer = BytePairTokenizer(vocab_size=1000)

# Load instead of training
tokenizer.load("tokenizer.pkl")

ids = tokenizer.encode("hello world")
```