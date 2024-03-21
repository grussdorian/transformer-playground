# A playground implementing a simple language model

### Model implements

1. Self attention
2. Multi headed attention
3. Decoder of the transformer block
4. Character level tokenization
5. Layer norms and dropouts added

|n_embed|n_head|n_layer|dropout|
|-------|------|-------|-------|
|384    | 6    | 6     | 0.2   |


Everything is implemented from scratch but layer norm used is pytorch's built in layer norm

- Based on the paper [attention is all you need]("https://arxiv.org/abs/1706.03762") and [Tutorial]("https://www.youtube.com/watch?v=kCc8FmEb1nY")
