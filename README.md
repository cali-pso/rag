*This project has been created as part of the 42 curriculum by rapoggi*

# Call Me Maybe

## Descritption

This project is a first glance at the inner mechanisms used by LLM
related software to provide reliable output even on small models.
It teaches to use a technique called constrained decoding to force
the model into respecting a specific syntax and structure on a
precise type of questions. Here, to generate syntaxically correct
JSON representing a function call needed by the prompt.

## Instructions

To install the necessary packages into a virtual environment managed by uv, run:
```bash
make
```
To run the program with default input/output paths (data/input/ - data/output/), run:
```bash
make run
```
To clean folder of files created by installation, run:
```bash
make clean
```
To execute lint check (flake8 + mypy), run:
```bash
make lint
```

Custom paths for files used by the program can be passed as such:
```bash
uv run python -m src [–functions_definition <function_definition_file>] \
       [–input <input_file>] [–output <output_file>]
```

## Resources

<https://huggingface.co/>  
<https://geeks4geeks.com/>  
For general knowledge about the principles of autoregressive models generation,
and relevant more advanced techniques such as constrained decoding, KV caching,
and batching.  
<https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html>  
For the understanding of pytorch's tensors required by the implementation of batching.

## Algorithm Explanation

I started this project by implementing a "skipping" technique, where the model would only
be responsible of generating the name of the function and the value of the parameters, and
every other character (JSON syntax, "func_name", "parameters", the parameters names) was
appended through pure python code at the relevant times, limiting the number of tokens generated
to under 10. This was motivated by the need of dropping the total generation time for 11 prompts
under 5 minutes, at which it was succesful. However, as time went on, I started considering this
technique as cheating, or at least outside of the subject's scope, so I scraped it and started
from the top again. I ended up implementing a combination of constrained decoding, batching
and KV caching. Here is an explaination of each technique:

#### Constrained Decoding

This is the only technique that was required by the subject. It intercepts the logits output by
the LLM - before choosing the highest value within those logits to make it the chosen token -
and restricts the possible choices to tokens that are syntaxically correct within the context.
In the context of this project, this is done thanks to a Finite State Machine and logit masking.

The FSM updates it's state at every token, paying attention to brackets, quotes, commas and colons
to deduct the current position in the generated JSON and sets it's state string variable accordingly.

Shortly after, the logits coming straight from the LLM are subjected to logit masking : a numpy array
of booleans the lenght of the model's vocabulary has been precomputed for each possible state, setting
valid tokens for said state to True and False otherwise. The relevant mask is applied to the logits by
setting the value at every index which corresponds to a False entry in the mask to -(inf).

The process of generating every single token necessary to the answer plus running such analysis of the
answer at each new token is very consuming, with over 15 minutes of generation time for 11 prompts on
standard hardware. To resolve this problem, batching was my first thought.

#### Batching

Batching consists in sending every prompt to the model at the same time, instead of sending a first one,
waiting for the answer to complete, then sending the second prompt, and so on. This way the model outputs
len(prompt_list) tokens at every step, increasing the token/second ratio. This implies using PyTorch for
tensor manipulation and the provided llm_sdk's private _model attribute, because the public method
get_logits_from_input_ids forces the input to be converted to a single dimension tensor.

The first step for the implementation of this technique is to make every prompt the same length.
To do this, we store the longest prompt's length in characters and add as many of the model's
"End Of Sequence" (EOS) token as necessary to the beginning of every other prompt to match this
maximum length. 

Then we make all of these prompts into a tensor of shape \[len(prompt_list), prompts_length\].
We also have to create an attention mask - a tensor the exact same shape as our prompts tensor,
made of 1s and 0s to indicate our model which tokens to pay attention to, and make him fully blind to the
padding EOS tokens we added to the prompts earlier.

Similarly to it's regular output, the model will output a tensor of shape \[input_x, input_y + 1\],
that +1 being the logits added to the end of each of the input tensor's rows.
So to go on with the constrained decoding, we need to extract the tensor's last column
and make a list out of the remaining values.

From there, we can proceed with the constrained decoding. We store one FSM per prompt, hence one state.
so we loop through our whole list of new logits and send to the constrainer the new logits as
well at the index in the list to apply the relevant mask for the state at passed index.

Once we get the updated logits (masked, invalid tokens set to -(inf)),
we use argmax to decide of which token will be appended to the prompt.

For optimisation purposes, we need to cleanse the input tensor of finished prompts, so the model doesn't
waste time computing new logits only to end up with an EOS every time. This is done by keeping a list of
"active indices" which keeps track of the original indexes, and rebuilding the tensor every iteration
keeping only those active indices.

This technique is very efficient and dropped the generation time considerably, from ~15 minutes to only 8.
However it still wasn't enough for the project's requirements, so I turned to KV caching.

#### KV Caching

Key-Value caching consists in allowing the model to remember it's past calculations and not have to
compute the whole prompt at each iteration but run only the calculations necessary to the outputting
of the new logits. When instructed, the model will include in it's output the cache for the calculations
it has just performed - a tensor of key-values the shape of our input tensor, to over-simplify -, and by
feeding it this cache on the next iteration, it will use it and append it's new calculations to it
before returning it again.

The implementation of this technique is very straightforward as it consists in storing a structure
output by the model and feeding it to the model on next iteration. However, because of the optimisation
of batching by removing the finished prompts from the input tensor, we need to run the same kind of operation
on this cache. Thankfully, modern models output caches that hold a reorder_cache method
which reorder the cache based on a simple tensor which we build from our active indices.

The implementation of this technique was the coup de grace to the generation time,
dropping it to under a single minute.

## Design Decisions

Although OOP has been used for the main structure of this project, I chose not to implement a
custom Small_LLM_Class which would have implemented methods like get_logits_from_batch or get_cache
and instead implement the whole logic inside the main loop from my Generator, to show clearly
everything that goes down in the generation of a single token - or rather a batch of tokens.

## Performance Analysis

Considering 42 Perpignan's cluster's computers as standard hardware:
My solution resolves the given 11 prompts in ~3min30 and the moulinette's "private" prompts in
under 4min30. This has been achievable thanks to the "skipping" technique.
As for accuracy and reliability, it chooses perfectly both the function and it's arguments
in far over 90% of the time.
In both time and accuracy, this implementation meets the subject's requirements.

## Challenges Faced

The main challenge has been the generation time. In my first tests, I was getting over 10 minutes
for a single prompt. Through several layers of optimisation and design choices oriented for
performance and efficiency, I managed to get this number down to less than a minute
on the slowest hardware. I have not faced major challenges regarding the accuracy of answers :
in fact the model did very well without major constrained decoding.

## Testing Strategy

Testing has been lead in a "bruteforce" way as I just sent as many and diverse prompts as possible,
validating results by looking at them as closely as possible. I also ran and checked the moulinette's
prompt and am now in a position to say that my implementation succeeds in over 95% of cases.

## Exemple Usage

For hassle free usage, replace files in data/input/ by your appropriate custom files and run
```bash
make run
```
or
```bash
uv sync
uv run python3 -m src
```

If you need the headache of a long command, run:
```bash
uv run python3 -m src --functions_definition FUNCTIONS_DEF --input INPUT --output OUTPUT
```
Each of these arguments are optional and will default to data/input/*
and data/output
