def preprocess(input_text, tokenizer):
    
        return tokenizer.encode_plus(
                            input_text,
                            add_special_tokens = True,
                            max_length = 32,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                        )