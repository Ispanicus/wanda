# Import necessary modules
from datasets import load_dataset
import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

import time
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

def eval_belebele(model, tokenizer, BATCH_SIZE=4, quantized=False):
    print(f"evaluating on belebele")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    seed_value = 42 
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed_value)

    dataset = load_dataset("facebook/belebele", "spa_Latn", split='test')
    eos_token_id = tokenizer.eos_token_id

    answers = dict()
    num_rows = dataset.num_rows
    input_texts = []

    for i in tqdm(range(num_rows)):
        passage = dataset['flores_passage'][i]
        question = dataset['question'][i]
        answer_a = dataset['mc_answer1'][i]
        answer_b = dataset['mc_answer2'][i]
        answer_c = dataset['mc_answer3'][i]
        answer_d = dataset['mc_answer4'][i]

        input_text = f'''
        Lee el siguiente texto y responde a la pregunta.

        Texto:
        {passage}

        Pregunta:
        {question}

        A) {answer_a}
        B) {answer_b}
        C) {answer_c}
        D) {answer_d}

        ¿De las cuatro opciones (A, B, C o D), cuál es la respuesta correcta?

        Respuesta: '''

        input_texts.append(input_text)

        if (i+1) % BATCH_SIZE == 0 or i == num_rows - 1:
            input_ids = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True)['input_ids'].to(device)

            output_ids_batch = model.generate(
                input_ids,
                max_length=max([len(ids) for ids in input_ids]) + 20,  # adjust for each batch
                num_return_sequences=1,
                top_k=1, # greedy sampling
                temperature=1,
                eos_token_id=eos_token_id,
                early_stopping=True,
            )

            for j, output_ids in enumerate(output_ids_batch):
                generated_tokens = output_ids[len(input_ids[j]):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers[i - BATCH_SIZE + j + 1] = (generated_text, dataset['correct_answer_num'][i - BATCH_SIZE + j + 1])

            # Reset input_texts for the next batch
            input_texts = []

    return answers

def eval_xquad(model, tokenizer, BATCH_SIZE=4, quantized=False):
    print(f"evaluating on XQUAD")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    seed_value = 42 
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed_value)

    dataset = load_dataset("xquad", "xquad.es", split='validation')
    eos_token_id = tokenizer.eos_token_id

    answers = dict()
    num_rows = dataset.num_rows
    input_texts = []

    for i in tqdm(range(num_rows)):
        passage = dataset['context'][i]
        question = dataset['question'][i]

        input_text = f'''
        Lee el siguiente texto y responde a la pregunta de manera concisa y breve.

        Texto:
        {passage}

        Pregunta:
        {question}

        Respuesta breve: '''

        input_texts.append(input_text)

        if (i+1) % BATCH_SIZE == 0 or i == num_rows - 1:
            input_ids = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True)['input_ids'].to(device)

            output_ids_batch = model.generate(
                input_ids,
                max_length=max([len(ids) for ids in input_ids]) + 30,  # adjust for each batch
                num_return_sequences=1,
                top_k=1, # greedy sampling
                temperature=1,
                eos_token_id=eos_token_id,
                early_stopping=True,
            )

            for j, output_ids in enumerate(output_ids_batch):
                generated_tokens = output_ids[len(input_ids[j]):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers[dataset['id'][i - BATCH_SIZE + j + 1]] = (generated_text, dataset['answers'][i - BATCH_SIZE + j + 1])

            # Reset input_texts for the next batch
            input_texts = []

    return answers

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "Spanish wikipedia"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    trainloader, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
        ppl_train = eval_ppl_wikitext_train(model, trainloader, 1, device)
    return ppl_train, ppl_test

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()