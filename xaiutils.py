import glob
from pathlib import Path


import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import softmax
import shap
from lime.lime_text import LimeTextExplainer
import torch
from captum.attr import IntegratedGradients


MAX_LENGTH = 512
MAX_EVALS  = 100
np.random.seed(42)


def predict_fn(tokenizer, model, texts):
    inputs = tokenizer(
        [str(t) for t in texts],
        return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()


def occlusion(tokenizer, model, text, masking="Mask"):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    input_ids      = inputs.input_ids[0]
    original_probs = predict_fn(tokenizer, model, [text])
    predicted_idx  = np.argmax(original_probs)
    mask_token     = tokenizer.mask_token_id if masking == "Mask" else tokenizer.pad_token_id

    variations = []
    for i in range(1, input_ids.shape[0] - 1):
        masked_tokens       = inputs.input_ids.clone()
        masked_tokens[0][i] = mask_token
        masked_sentence     = tokenizer.decode(masked_tokens[0], skip_special_tokens=True)
        probs               = predict_fn(tokenizer, model, [masked_sentence])
        delta               = abs(original_probs[0][predicted_idx] - probs[0][predicted_idx])
        variations.append(delta)

    return variations / np.sum(variations)


def shap_explanation(tokenizer, model, instance):
    masker       = shap.maskers.Text(tokenizer)
    _predict     = lambda x: predict_fn(tokenizer, model, x)
    explainer_sh = shap.Explainer(_predict, masker=masker, output_names=["NEG", "POS"])
    shap_vals    = explainer_sh([instance], max_evals=MAX_EVALS)
    predicted_class = np.argmax(predict_fn(tokenizer, model, [instance]))
    shap_values     = shap_vals.values.squeeze()[:, predicted_class]
    return np.abs(shap_values[1:len(shap_vals[0]) - 1])


def lime_explanation(tokenizer, model, instance):
    explainer = LimeTextExplainer(class_names=["neg", "pos"], bow=False)
    _predict  = lambda x: predict_fn(tokenizer, model, x)
    exp = explainer.explain_instance(
        instance, _predict,
        num_features=1000,
        num_samples=1000,
    )
    exp_map           = exp.as_map()
    pairs             = exp_map[1]
    indexed_string    = exp.domain_mapper.indexed_string
    importance_vector = np.zeros(indexed_string.num_words())
    for idx, val in pairs:
        importance_vector[idx] = val
    return np.abs(importance_vector)


def build_baseline(
    tokenizer,
    model,
    input_ids,
    method="pad",
    random_texts=None
):
    """
    Returns baseline embeddings according to masking strategy
    """
    device = model.device
    emb_layer = model.get_input_embeddings()

    if method == "pad":
        baseline_ids = torch.full_like(
            input_ids,
            tokenizer.pad_token_id
        )

    elif method == "cls":
        baseline_ids = torch.full_like(
            input_ids,
            tokenizer.cls_token_id
        )

    elif method == "sep":
        baseline_ids = torch.full_like(
            input_ids,
            tokenizer.sep_token_id
        )

    elif method == "mask":
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no mask token.")
        baseline_ids = torch.full_like(
            input_ids,
            tokenizer.mask_token_id
        )

    elif method == "unk":
        if tokenizer.unk_token_id is None:
            raise ValueError("Tokenizer has no unk token.")
        baseline_ids = torch.full_like(
            input_ids,
            tokenizer.unk_token_id
        )

    elif method == "zero":
        return torch.zeros_like(
            emb_layer(input_ids)
        ).to(device)

    elif method == "random_tokens":
        vocab_size = tokenizer.vocab_size

        baseline_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=input_ids.shape,
            device=device
        )

    elif method == "random_text":
        if random_texts is None:
            raise ValueError("Provide random_texts.")

        rand_text = np.random.choice(random_texts)

        baseline_inputs = tokenizer(
            rand_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=input_ids.shape[1]
        ).to(device)

        baseline_ids = baseline_inputs.input_ids

    else:
        raise ValueError(f"Unknown masking method: {method}")

    return emb_layer(baseline_ids)


def build_baseline_from_disk(model, path) -> iter:
    """
    Creates a memory-efficient generator that yields baseline embeddings 
    by sequentially loading input_ids.pt files from the specified directory.
    Avoids loading all baselines into RAM simultaneously.
    """
    emb_layer = model.get_input_embeddings()
    device = model.device
 
    # Locate all .pt files recursively within the provided path
    pt_files = sorted(glob.glob(str(Path(path).resolve()) + "/**/input_ids.pt", recursive=True))

    if not pt_files:
        raise ValueError(f"No .pt files found at the specified path: {path}")

    for pt_file in pt_files:
        input_ids = torch.load(pt_file, map_location=device, weights_only=False)

        # TODO: check if this is necessary or not
        # Ensure consistent batch dimension for embedding lookup
        # if input_ids.dim() == 1:
            # input_ids = input_ids.unsqueeze(0)

        # Yield only the embedding representation, deferring computation/storage
        yield emb_layer(input_ids)


def integrated_gradients_explanation(
    tokenizer,
    model,
    text,
    masking_method="pad",
    random_texts=None,
    baseline_emb=None,
    max_length=128,
    n_steps=50
):
    """
    Token-level Integrated Gradients explanation.
    Accepts predefined baselines when masking_method is 'from_disk'.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    ).to(model.device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    emb_layer = model.get_input_embeddings()
    embeddings = emb_layer(input_ids)

    if masking_method != "from_disk":
        baseline_emb = build_baseline(
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            method=masking_method,
            random_texts=random_texts
        )
    elif masking_method == "from_disk" and baseline_emb is None:
        raise ValueError("baseline_emb must be provided when masking_method is 'from_disk'.")

    def forward_fn(embeds, attention_mask):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs.logits

    with torch.no_grad():
        logits = model(**inputs).logits
        target = torch.argmax(logits, dim=1)

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(
        embeddings,
        baselines=baseline_emb,
        additional_forward_args=(attention_mask,),
        target=target,
        n_steps=n_steps
    )

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.detach().cpu().numpy()
    attributions = np.abs(attributions) / np.sum(np.abs(attributions))
    return attributions[1:-1]


def expected_gradients_explanation(
    tokenizer,
    model,
    text,
    masking_method="random_text",
    random_texts=None,
    baselines_path=None,
    n_baselines=20,
    max_length=128,
    n_steps=50
):
    """
    Expected Gradients: computes the mean attribution over multiple baseline runs.
    Supports both on-the-fly generation and streaming from disk.
    """
    all_attr = []

    if masking_method == "from_disk":
        if baselines_path is None:
            raise ValueError("baselines_path must be provided when masking_method is 'from_disk'.")

        # Determine effective token count for the input sequence.
        # Subtract 2 to exclude the initial and final special tokens (BOS/EOS).
        tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        seq_len = tokenized.input_ids.size(1)
        n_tokens = max(1, seq_len - 2)

        # Resolve baseline directory based on token count.
        baseline_dir = Path(baselines_path) / f"{n_tokens}_tokens"
        if not baseline_dir.exists():
            raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

        baseline_gen = build_baseline_from_disk(model, baseline_dir)
        processed_count = 0

        # Iterate through the generator, respecting the baseline limit
        while True:
            try:
                baseline_emb = next(baseline_gen)
                attr = integrated_gradients_explanation(
                    tokenizer=tokenizer,
                    model=model,
                    text=text,
                    masking_method="from_disk",
                    baseline_emb=baseline_emb,
                    max_length=max_length,
                    n_steps=n_steps
                )
                all_attr.append(attr)
                processed_count += 1

                # Stop if limit is reached (n_baselines > 0). -1 means process all available files.
                if n_baselines != -1 and processed_count >= n_baselines:
                    break
            except StopIteration:
                break

    else:
        for _ in range(n_baselines):
            attr = integrated_gradients_explanation(
                tokenizer=tokenizer,
                model=model,
                text=text,
                masking_method=masking_method,
                random_texts=random_texts,
                max_length=max_length,
                n_steps=n_steps
            )
            all_attr.append(attr)

    return np.mean(all_attr, axis=0)
